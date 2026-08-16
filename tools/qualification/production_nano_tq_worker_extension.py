# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Qualification-only Megatron worker with durable optimizer evidence.

This class is deliberately additive.  Production training workers keep their
normal implementation; the fixed-batch qualification runner selects this
extension explicitly through ``worker_extension_cls_fqn``.

The journal provides an at-most-once boundary, not crash recovery.  A worker
writes ``optimizer-dispatched`` before entering the production
``train_presharded`` implementation and ``optimizer-applied`` only after it
returns and a post-update parameter/gradient digest has been collected.  If a
worker dies or the Ray acknowledgement is lost between those records, the
outcome is ambiguous and the run must not be replayed automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import ray
import torch

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


JOURNAL_FORMAT = "nemo-rl-production-nano-tq-optimizer-journal-v1"
PARAMETER_DIGEST_FORMAT = "nemo-rl-local-trainable-parameter-bytes-v1"
SAFE_RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NCCL_ALGO",
    "NCCL_CROSS_NIC",
    "NCCL_DEBUG",
    "NCCL_IB_HCA",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_LEVEL",
    "NCCL_PROTO",
    "NCCL_SOCKET_IFNAME",
    "RAY_ADDRESS",
    "RAY_NAMESPACE",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
)
SECRET_ENV_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NGC_API_KEY",
    "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "WANDB_API_KEY",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> str:
    """Create one durable, immutable journal record and return its digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # fdopen owns the descriptor after it succeeds.  A partial exclusive
        # record is intentionally retained as corruption evidence; callers
        # must start a fresh qualification run rather than overwrite it.
        raise
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(encoded).hexdigest()


def _resolved_module_path(module: Any) -> str:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"module {module.__name__!r} has no file identity")
    return str(Path(value).resolve(strict=True))


def _mountinfo_record(path: Path, *, expected_read_only: bool) -> dict[str, Any]:
    """Record the most-specific Linux mount and enforce its access mode."""
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise RuntimeError("/proc/self/mountinfo is unavailable in worker")

    def decode(value: str) -> str:
        for encoded, decoded in (
            ("\\040", " "),
            ("\\011", "\t"),
            ("\\012", "\n"),
            ("\\134", "\\"),
        ):
            value = value.replace(encoded, decoded)
        return value

    resolved = path.resolve(strict=True)
    candidates: list[tuple[Path, list[str], list[str], str, str]] = []
    for raw_line in mountinfo.read_text(encoding="utf-8").splitlines():
        left, separator, right = raw_line.partition(" - ")
        if not separator:
            raise RuntimeError("malformed /proc/self/mountinfo record")
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            raise RuntimeError("truncated /proc/self/mountinfo record")
        mount_point = Path(decode(left_fields[4]))
        if resolved == mount_point or resolved.is_relative_to(mount_point):
            candidates.append(
                (
                    mount_point,
                    left_fields[5].split(","),
                    right_fields[2].split(","),
                    right_fields[0],
                    decode(right_fields[1]),
                )
            )
    if not candidates:
        raise RuntimeError(f"no mountinfo record covers worker path {resolved}")
    mount_point, mount_options, super_options, fs_type, source = max(
        candidates, key=lambda item: len(item[0].parts)
    )
    observed_read_only = "ro" in mount_options and "rw" not in mount_options
    if observed_read_only != expected_read_only:
        raise RuntimeError(
            f"worker mount mode changed for {resolved}: options={mount_options}, "
            f"expected_read_only={expected_read_only}"
        )
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mount_point": str(mount_point),
        "mount_options": mount_options,
        "super_options": super_options,
        "filesystem_type": fs_type,
        "source": source,
        "read_only": observed_read_only,
    }


def _auto_dependency_provenance(
    module: Any,
    *,
    source_root: Path,
) -> dict[str, str]:
    """Resolve a module to its one interpreter-selected owning distribution."""
    top_level_name = module.__name__.split(".", 1)[0]
    owner_names = sorted(set(metadata.packages_distributions().get(top_level_name, [])))
    module_path = Path(_resolved_module_path(module))
    matches: list[tuple[str, Any, Path]] = []
    for owner_name in owner_names:
        distribution = metadata.distribution(owner_name)
        anchor = Path(
            distribution.locate_file(Path(top_level_name) / module_path.name)
        ).resolve(strict=True)
        if anchor == module_path:
            matches.append((owner_name, distribution, anchor))
    if len(matches) != 1:
        raise RuntimeError(
            f"dependency owner is ambiguous for {module.__name__}: "
            f"owners={owner_names}, matching={[(name, str(path)) for name, _, path in matches]}"
        )
    owner_name, distribution, anchor_path = matches[0]
    if module_path.is_relative_to(source_root):
        raise RuntimeError(
            f"runtime dependency came from source mount: "
            f"{module.__name__}={module_path}"
        )
    return {
        "module_path": str(module_path),
        "distribution_name": owner_name,
        "distribution_anchor": str(anchor_path),
        "distribution_root": str(Path(distribution.locate_file("")).absolute()),
        "distribution_version": distribution.version,
    }


def _dependency_provenance(
    module: Any,
    distribution_name: str,
    *,
    source_root: Path,
) -> dict[str, str]:
    """Bind an import to the exact package anchor selected by this interpreter."""
    distribution = metadata.distribution(distribution_name)
    module_path = Path(_resolved_module_path(module))
    top_level_name = module.__name__.split(".", 1)[0]
    anchor_relative = Path(top_level_name) / module_path.name
    anchor_path = Path(distribution.locate_file(anchor_relative)).resolve(strict=True)
    if module_path != anchor_path:
        raise RuntimeError(
            "runtime dependency does not match its interpreter-selected "
            f"distribution anchor: {module.__name__}={module_path}, "
            f"anchor={anchor_path}"
        )
    if module_path.is_relative_to(source_root):
        raise RuntimeError(
            f"runtime dependency came from source mount: "
            f"{module.__name__}={module_path}"
        )
    return {
        "module_path": str(module_path),
        "distribution_anchor": str(anchor_path),
        "distribution_root": str(Path(distribution.locate_file("")).absolute()),
        "distribution_version": distribution.version,
    }


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover - exercised only by the production Ray qualification
class ProductionNanoTQMegatronPolicyWorker(MegatronPolicyWorkerImpl):
    """Megatron worker that adds bounded provenance and optimizer evidence."""

    _qualification_state: dict[str, Any] | None = None
    _qualification_runtime: dict[str, str] | None = None

    def _rank_identity(self) -> dict[str, Any]:
        rank = int(torch.distributed.get_rank())
        return {
            "rank": rank,
            "world_size": int(torch.distributed.get_world_size()),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "local_coords": self._local_coords(),
        }

    def qualification_debug_event(
        self,
        *,
        run_id: str,
        debug_root: str,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one immutable per-rank stage event without dumping the env."""
        if not run_id or not stage or status not in {"started", "completed", "failed"}:
            raise ValueError("invalid qualification actor debug identity")
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in stage):
            raise ValueError(f"invalid qualification actor stage: {stage!r}")
        root = Path(debug_root).resolve(strict=True)
        expected_root = os.environ.get("NRL_QUALIFICATION_DEBUG_ROOT")
        if expected_root is None or Path(expected_root).resolve(strict=True) != root:
            raise RuntimeError("actor debug root differs from scheduler authority")
        image_digest = os.environ.get("NRL_QUALIFIED_IMAGE_DIGEST")
        if not image_digest:
            raise RuntimeError("actor has no qualified image digest")
        self._qualification_runtime = {
            "run_id": run_id,
            "debug_root": str(root),
            "image_digest": image_digest,
        }
        identity = self._rank_identity()
        payload = {
            "format": "nemo-rl-production-nano-tq-actor-stage-v1",
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "time_ns": time.time_ns(),
            "image_digest": image_digest,
            **identity,
            "details": dict(details or {}),
        }
        path = (
            root
            / f"rank-{identity['rank']:05d}"
            / f"{payload['time_ns']}-{stage}-{status}.json"
        )
        record_sha256 = _write_json_exclusive(path, payload)
        return {
            "run_id": run_id,
            **identity,
            "stage": stage,
            "status": status,
            "record_path": str(path),
            "record_sha256": record_sha256,
        }

    def _runtime_debug_event(
        self,
        *,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        runtime = self._qualification_runtime
        if runtime is None:
            raise RuntimeError("qualification actor debug runtime is not initialized")
        self.qualification_debug_event(
            run_id=runtime["run_id"],
            debug_root=runtime["debug_root"],
            stage=stage,
            status=status,
            details=details,
        )

    def qualification_provenance(
        self,
        *,
        run_id: str,
        expected_source_root: str,
        expected_model_root: str,
        expected_run_root: str,
        expected_actor_venv: str,
        expected_image_digest: str,
        expected_python_version: str,
        expected_ray_version: str,
        expected_torch_version: str,
        expected_cuda_compute_capability: str,
    ) -> dict[str, Any]:
        """Return fail-closed source/dependency provenance for this actor."""
        import megatron.bridge
        import megatron.core
        import mooncake
        import mooncake.store
        import nemo_rl
        import nemo_rl.data_plane.packed_tensor_wire as packed_tensor_wire
        import tensordict
        import transformer_engine
        import transformer_engine.pytorch
        import transfer_queue

        from nemo_rl.data_plane.adapters.transfer_queue import (
            validate_baked_transfer_queue,
        )

        if not run_id:
            raise ValueError("qualification run_id must be non-empty")
        source_root = Path(expected_source_root).resolve(strict=True)
        model_root = Path(expected_model_root).resolve(strict=True)
        run_root = Path(expected_run_root).resolve(strict=True)
        actor_venv = Path(expected_actor_venv).resolve(strict=True)
        if Path(sys.prefix).resolve(strict=True) != actor_venv:
            raise RuntimeError(
                f"wrong actor venv: prefix={sys.prefix}, expected={actor_venv}"
            )
        if (
            not sys.dont_write_bytecode
            or not sys.flags.no_user_site
            or not sys.flags.safe_path
        ):
            raise RuntimeError(
                "actor must use no-bytecode, no-user-site, and safe-path flags"
            )
        observed_image_digest = os.environ.get("NRL_QUALIFIED_IMAGE_DIGEST")
        if observed_image_digest != expected_image_digest:
            raise RuntimeError(
                "actor image identity changed: "
                f"{observed_image_digest!r} != {expected_image_digest!r}"
            )
        python_version = ".".join(str(item) for item in sys.version_info[:3])
        if python_version != expected_python_version:
            raise RuntimeError(
                f"actor Python changed: {python_version} != {expected_python_version}"
            )
        if ray.__version__ != expected_ray_version:
            raise RuntimeError(
                f"actor Ray changed: {ray.__version__} != {expected_ray_version}"
            )
        torch_release = torch.__version__.split("+", 1)[0]
        if torch_release != expected_torch_version:
            raise RuntimeError(
                f"actor Torch changed: {torch.__version__} != {expected_torch_version}"
            )
        cuda_compute_capability = ".".join(
            str(item) for item in torch.cuda.get_device_capability()
        )
        if cuda_compute_capability != expected_cuda_compute_capability:
            raise RuntimeError(
                "actor CUDA compute capability changed: "
                f"{cuda_compute_capability} != {expected_cuda_compute_capability}"
            )

        expected_source_paths = {
            "nemo_rl": source_root / "nemo_rl" / "__init__.py",
            "nemo_rl.data_plane.packed_tensor_wire": (
                source_root / "nemo_rl" / "data_plane" / "packed_tensor_wire.py"
            ),
            "megatron.bridge": (
                source_root
                / "3rdparty"
                / "Megatron-Bridge-workspace"
                / "Megatron-Bridge"
                / "src"
                / "megatron"
                / "bridge"
                / "__init__.py"
            ),
            "megatron.core": (
                source_root
                / "3rdparty"
                / "Megatron-Bridge-workspace"
                / "Megatron-Bridge"
                / "3rdparty"
                / "Megatron-LM"
                / "megatron"
                / "core"
                / "__init__.py"
            ),
        }
        source_modules = (nemo_rl, packed_tensor_wire, megatron.bridge, megatron.core)
        source_paths = {}
        for module in source_modules:
            module_path = Path(_resolved_module_path(module))
            expected_path = expected_source_paths[module.__name__].resolve(strict=True)
            if module_path != expected_path:
                raise RuntimeError(
                    f"source module path changed: {module.__name__}={module_path}, "
                    f"expected={expected_path}"
                )
            source_paths[module.__name__] = str(module_path)
        dependency_paths = {
            module.__name__: _dependency_provenance(
                module,
                distribution_name,
                source_root=source_root,
            )
            for module, distribution_name in (
                (torch, "torch"),
                (ray, "ray"),
                (transfer_queue, "TransferQueue"),
                (tensordict, "tensordict"),
                (mooncake, "mooncake-transfer-engine-cuda13"),
            )
        }
        dependency_paths[transformer_engine.__name__] = (
            _auto_dependency_provenance(
                transformer_engine,
                source_root=source_root,
            )
        )
        transfer_queue_provenance = validate_baked_transfer_queue()
        runtime_context = ray.get_runtime_context()
        nccl_version = torch.cuda.nccl.version()
        if isinstance(nccl_version, tuple):
            nccl_version = list(nccl_version)

        return {
            "format": "nemo-rl-production-nano-tq-actor-provenance-v1",
            "run_id": run_id,
            **self._rank_identity(),
            "actor_venv": str(actor_venv),
            "executable": str(Path(sys.executable).resolve(strict=True)),
            "python_flags": {
                "dont_write_bytecode": bool(sys.dont_write_bytecode),
                "no_user_site": bool(sys.flags.no_user_site),
                "safe_path": bool(sys.flags.safe_path),
            },
            "source_root": str(source_root),
            "model_root": str(model_root),
            "run_root": str(run_root),
            "source_modules": source_paths,
            "dependency_modules": dependency_paths,
            "transformer_engine_pytorch_module": _resolved_module_path(
                transformer_engine.pytorch
            ),
            "mooncake_store_module": _resolved_module_path(mooncake.store),
            "python_version": python_version,
            "ray_version": ray.__version__,
            "ray_runtime": {
                "actor_id": str(runtime_context.get_actor_id()),
                "node_id": str(runtime_context.get_node_id()),
                "namespace": runtime_context.namespace,
            },
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_nccl_version": nccl_version,
            "cuda_compute_capability": cuda_compute_capability,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_total_memory": torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory,
            "image_digest": expected_image_digest,
            "mounts": {
                "source": _mountinfo_record(
                    source_root, expected_read_only=True
                ),
                "model": _mountinfo_record(model_root, expected_read_only=True),
                "output": _mountinfo_record(run_root, expected_read_only=False),
            },
            "safe_runtime_env": {
                name: os.environ.get(name) for name in SAFE_RUNTIME_ENV_KEYS
            },
            "secret_env_presence": {
                name: bool(os.environ.get(name)) for name in SECRET_ENV_KEYS
            },
            "transfer_queue": transfer_queue_provenance,
        }

    @staticmethod
    def _meta_debug_identity(meta: Any) -> dict[str, Any]:
        schema = (meta.extra_info or {}).get("nrl_packed_tensor_wire_v1") or {}
        return {
            "partition_id": meta.partition_id,
            "sample_count": len(meta.sample_ids),
            "sample_ids_sha256": hashlib.sha256(
                _canonical_json(list(meta.sample_ids))
            ).hexdigest(),
            "media_wire_schema_id": schema.get("wire_schema_id"),
            "task_name": meta.task_name,
        }

    def get_logprobs_presharded(
        self,
        meta: Any,
        micro_batch_size: int | None = None,
    ) -> None:
        """Record rank-local prev-logprob boundaries around the real path."""
        details = self._meta_debug_identity(meta)
        self._runtime_debug_event(
            stage="prev-lp", status="started", details=details
        )
        try:
            result = super().get_logprobs_presharded(
                meta,
                micro_batch_size=micro_batch_size,
            )
        except BaseException as error:
            self._runtime_debug_event(
                stage="prev-lp",
                status="failed",
                details={**details, "error_type": type(error).__name__},
            )
            raise
        self._runtime_debug_event(
            stage="prev-lp", status="completed", details=details
        )
        return result

    def get_reference_policy_logprobs_presharded(
        self,
        meta: Any,
        micro_batch_size: int | None = None,
    ) -> None:
        """Record rank-local reference-logprob boundaries around the real path."""
        details = self._meta_debug_identity(meta)
        self._runtime_debug_event(
            stage="ref-lp", status="started", details=details
        )
        try:
            result = super().get_reference_policy_logprobs_presharded(
                meta,
                micro_batch_size=micro_batch_size,
            )
        except BaseException as error:
            self._runtime_debug_event(
                stage="ref-lp",
                status="failed",
                details={**details, "error_type": type(error).__name__},
            )
            raise
        self._runtime_debug_event(
            stage="ref-lp", status="completed", details=details
        )
        return result

    @staticmethod
    def _update_tensor_bytes(
        digest: "hashlib._Hash",
        tensor: torch.Tensor,
        *,
        chunk_elements: int,
    ) -> int:
        value = tensor.detach().reshape(-1)
        for start in range(0, value.numel(), chunk_elements):
            chunk = (
                value[start : start + chunk_elements]
                .to(device="cpu", non_blocking=False)
                .contiguous()
            )
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return value.numel() * value.element_size()

    def _local_parameter_state(self, *, include_gradients: bool) -> dict[str, Any]:
        """Hash all local trainable parameter bytes with bounded host memory."""
        parameter_digest = hashlib.sha256()
        gradient_digest = hashlib.sha256()
        parameter_count = 0
        parameter_numel = 0
        parameter_bytes = 0
        gradient_count = 0
        gradient_numel = 0
        gradient_bytes = 0
        gradients_finite = True
        gradients_nonzero = False
        chunk_elements = 4 * 1024 * 1024

        for name, parameter in sorted(self.model.named_parameters()):
            if not parameter.requires_grad:
                continue
            header = _canonical_json(
                {
                    "name": name,
                    "dtype": str(parameter.dtype),
                    "shape": list(parameter.shape),
                }
            )
            parameter_digest.update(len(header).to_bytes(8, "big"))
            parameter_digest.update(header)
            parameter_bytes += self._update_tensor_bytes(
                parameter_digest,
                parameter,
                chunk_elements=chunk_elements,
            )
            parameter_count += 1
            parameter_numel += parameter.numel()

            if not include_gradients:
                continue
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            if gradient is None:
                continue
            gradient_header = _canonical_json(
                {
                    "name": name,
                    "dtype": str(gradient.dtype),
                    "shape": list(gradient.shape),
                }
            )
            gradient_digest.update(len(gradient_header).to_bytes(8, "big"))
            gradient_digest.update(gradient_header)
            gradient_bytes += self._update_tensor_bytes(
                gradient_digest,
                gradient,
                chunk_elements=chunk_elements,
            )
            gradient_count += 1
            gradient_numel += gradient.numel()
            gradients_finite = gradients_finite and bool(
                torch.isfinite(gradient).all().item()
            )
            gradients_nonzero = gradients_nonzero or bool(
                torch.count_nonzero(gradient).item()
            )

        if parameter_count == 0 or parameter_numel == 0:
            raise RuntimeError("qualification worker found no trainable parameters")
        if include_gradients and gradient_count == 0:
            raise RuntimeError("qualification worker found no materialized gradients")

        result: dict[str, Any] = {
            "format": PARAMETER_DIGEST_FORMAT,
            **self._rank_identity(),
            "parameter_count": parameter_count,
            "parameter_numel": parameter_numel,
            "parameter_bytes": parameter_bytes,
            "parameter_sha256": parameter_digest.hexdigest(),
        }
        if include_gradients:
            result.update(
                {
                    "gradient_count": gradient_count,
                    "gradient_numel": gradient_numel,
                    "gradient_bytes": gradient_bytes,
                    "gradient_sha256": gradient_digest.hexdigest(),
                    "gradients_finite": gradients_finite,
                    "gradients_nonzero": gradients_nonzero,
                }
            )
        return result

    def _journal_path(self, phase: str) -> Path:
        state = self._qualification_state
        if state is None:
            raise RuntimeError("qualification optimizer step is not armed")
        rank = int(torch.distributed.get_rank())
        return (
            Path(state["journal_root"])
            / f"rank-{rank:05d}"
            / f"{phase}.json"
        )

    def _write_phase(self, phase: str, fields: dict[str, Any]) -> str:
        state = self._qualification_state
        if state is None:
            raise RuntimeError("qualification optimizer step is not armed")
        payload = {
            "format": JOURNAL_FORMAT,
            "phase": phase,
            "run_id": state["run_id"],
            "step_id": state["step_id"],
            "partition_id": state["partition_id"],
            "sample_ids": state["sample_ids"],
            "media_wire_schema_id": state["media_wire_schema_id"],
            "time_ns": time.time_ns(),
            **self._rank_identity(),
            **fields,
        }
        return _write_json_exclusive(self._journal_path(phase), payload)

    def qualification_arm_optimizer_step(
        self,
        *,
        run_id: str,
        step_id: str,
        journal_root: str,
        partition_id: str,
        sample_ids: list[str],
        media_wire_schema_id: str,
    ) -> dict[str, Any]:
        """Capture the exact pre-update state and arm one optimizer dispatch."""
        if self._qualification_state is not None:
            raise RuntimeError("qualification optimizer step is already armed")
        root = Path(journal_root).resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError("qualification journal_root must be a real directory")
        if not run_id or not step_id or not sample_ids or len(set(sample_ids)) != len(
            sample_ids
        ):
            raise ValueError("invalid qualification optimizer identity")
        baseline = self._local_parameter_state(include_gradients=False)
        self._qualification_state = {
            "run_id": run_id,
            "step_id": step_id,
            "journal_root": str(root),
            "partition_id": partition_id,
            "sample_ids": list(sample_ids),
            "media_wire_schema_id": media_wire_schema_id,
            "baseline": baseline,
            "dispatched": False,
            "applied": False,
            "post_state": None,
        }
        record_sha256 = self._write_phase(
            "baseline",
            {"parameter_state": baseline},
        )
        return {
            "run_id": run_id,
            "step_id": step_id,
            **self._rank_identity(),
            "parameter_state": baseline,
            "record_sha256": record_sha256,
        }

    def train_presharded(self, meta: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Run the production train method behind a durable at-most-once guard."""
        state = self._qualification_state
        if state is None:
            raise RuntimeError("qualification train called before optimizer arm")
        if state["dispatched"]:
            raise RuntimeError(
                "qualification optimizer step was already dispatched; replay is forbidden"
            )
        if meta.partition_id != state["partition_id"]:
            raise RuntimeError("qualification partition identity changed")
        if list(meta.sample_ids) != state["sample_ids"]:
            raise RuntimeError("qualification sample identity changed")
        schema = (meta.extra_info or {}).get("nrl_packed_tensor_wire_v1") or {}
        if schema.get("wire_schema_id") != state["media_wire_schema_id"]:
            raise RuntimeError("qualification media wire identity changed")

        state["dispatched"] = True
        train_debug_details = self._meta_debug_identity(meta)
        train_debug_details["step_id"] = state["step_id"]
        self._runtime_debug_event(
            stage="train", status="started", details=train_debug_details
        )
        self._write_phase(
            "optimizer-dispatched",
            {
                "baseline_parameter_sha256": state["baseline"][
                    "parameter_sha256"
                ],
                "restart_safe_replay": False,
            },
        )
        try:
            result = super().train_presharded(meta, *args, **kwargs)
        except BaseException as error:
            self._runtime_debug_event(
                stage="train",
                status="failed",
                details={
                    **train_debug_details,
                    "error_type": type(error).__name__,
                    "optimizer_outcome": "ambiguous",
                },
            )
            self._write_phase(
                "optimizer-outcome-ambiguous",
                {
                    "error_type": type(error).__name__,
                    "restart_safe_replay": False,
                },
            )
            raise

        try:
            post_state = self._local_parameter_state(include_gradients=True)
        except BaseException as error:
            self._runtime_debug_event(
                stage="train-evidence",
                status="failed",
                details={
                    **train_debug_details,
                    "error_type": type(error).__name__,
                    "optimizer_outcome": "ambiguous",
                },
            )
            self._write_phase(
                "optimizer-outcome-ambiguous",
                {
                    "error_type": type(error).__name__,
                    "evidence_stage": "post-update-parameter-and-gradient-digest",
                    "restart_safe_replay": False,
                },
            )
            raise
        state["post_state"] = post_state
        state["applied"] = True
        changed = (
            post_state["parameter_sha256"]
            != state["baseline"]["parameter_sha256"]
        )
        if not changed:
            self._write_phase(
                "optimizer-applied-without-parameter-delta",
                {
                    "post_parameter_state": post_state,
                    "restart_safe_replay": False,
                },
            )
            raise RuntimeError("optimizer returned without a local parameter delta")
        if not post_state["gradients_finite"] or not post_state["gradients_nonzero"]:
            self._write_phase(
                "optimizer-applied-with-invalid-gradient-evidence",
                {
                    "post_parameter_state": post_state,
                    "restart_safe_replay": False,
                },
            )
            raise RuntimeError("optimizer returned without finite nonzero gradients")
        self._write_phase(
            "optimizer-applied",
            {
                "post_parameter_state": post_state,
                "parameter_delta": True,
                "restart_safe_replay": False,
            },
        )
        self._runtime_debug_event(
            stage="train",
            status="completed",
            details={
                **train_debug_details,
                "optimizer_outcome": "applied",
                "parameter_delta": True,
            },
        )
        return result

    def qualification_optimizer_evidence(self, *, run_id: str) -> dict[str, Any]:
        """Return the in-memory state after durable rank records were written."""
        state = self._qualification_state
        if state is None or state["run_id"] != run_id:
            raise RuntimeError("qualification optimizer identity is unavailable")
        if not state["dispatched"] or not state["applied"]:
            raise RuntimeError("qualification optimizer outcome is not applied")
        baseline = state["baseline"]
        post_state = state["post_state"]
        if post_state is None:
            raise RuntimeError("qualification post-update state is missing")
        return {
            "format": "nemo-rl-production-nano-tq-optimizer-evidence-v1",
            "run_id": run_id,
            "step_id": state["step_id"],
            **self._rank_identity(),
            "before_parameter_sha256": baseline["parameter_sha256"],
            "after_parameter_sha256": post_state["parameter_sha256"],
            "gradient_sha256": post_state["gradient_sha256"],
            "parameter_delta": (
                baseline["parameter_sha256"]
                != post_state["parameter_sha256"]
            ),
            "gradients_finite": post_state["gradients_finite"],
            "gradients_nonzero": post_state["gradients_nonzero"],
            "restart_safe_replay": False,
        }

    def qualification_record_checkpoint_join(
        self,
        *,
        run_id: str,
        checkpoint_tree_sha256: str,
        controller_result_sha256: str,
    ) -> dict[str, Any]:
        """Join a completed checkpoint tree to this rank's applied update."""
        state = self._qualification_state
        if state is None or state["run_id"] != run_id or not state["applied"]:
            raise RuntimeError("cannot join checkpoint before an applied optimizer step")
        if len(checkpoint_tree_sha256) != 64 or len(controller_result_sha256) != 64:
            raise ValueError("checkpoint join digests must be SHA256 hex strings")
        record_sha256 = self._write_phase(
            "checkpoint-joined",
            {
                "checkpoint_tree_sha256": checkpoint_tree_sha256,
                "controller_result_sha256": controller_result_sha256,
                "after_parameter_sha256": state["post_state"][
                    "parameter_sha256"
                ],
                "restart_safe_replay": False,
            },
        )
        return {
            "run_id": run_id,
            **self._rank_identity(),
            "checkpoint_tree_sha256": checkpoint_tree_sha256,
            "record_sha256": record_sha256,
        }


def verify_journal_file(path: Path) -> dict[str, Any]:
    """Small helper for controller-side rehash and dependency-free tests."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != JOURNAL_FORMAT:
        raise ValueError(f"unexpected journal format in {path}")
    if _sha256_file(path) != hashlib.sha256(
        _canonical_json(payload) + b"\n"
    ).hexdigest():
        raise ValueError(f"journal canonical bytes changed in {path}")
    return payload
