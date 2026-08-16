# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""One production Nano-Omni fixed-batch update through TransferQueue.

This qualification deliberately bypasses rollout generation, Gym, and vLLM.
It constructs a deterministic image/text batch with the *real* model processor
and then exercises the production training data path:

``prepare_step -> kv_first_write -> prev_lp -> ref_lp -> train -> checkpoint``

The first policy logprob result is copied into ``generation_logprobs`` before
training, making the fixed batch on-policy without inventing a second model
server.  The media wire authority is joined against worker R3 records for all
three consumers.  A qualification-only worker extension records full-byte
local parameter digests before and after the update and writes a durable
at-most-once optimizer journal.

Important boundary: an ``optimizer-dispatched`` record without a matching
``optimizer-applied`` record is ambiguous and MUST NOT be replayed.  This
harness detects that state; it does not make Megatron optimizer updates
transactional across process death.

The module keeps third-party imports inside runtime functions.  Its manifest,
state-machine, trace-join, and journal validators can therefore be tested with
the Python standard library in the macOS review checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULT_FORMAT = "nemo-rl-production-nano-tq-fixed-batch-v1"
RUN_INTENT_FORMAT = "nemo-rl-production-nano-tq-run-intent-v1"
CHECKPOINT_MANIFEST_FORMAT = "nemo-rl-production-nano-tq-checkpoint-tree-v1"
WORKER_EXTENSION_FQN = (
    "tools.qualification.production_nano_tq_worker_extension."
    "ProductionNanoTQMegatronPolicyWorker"
)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUIRED_STAGES = ("prev_lp", "ref_lp", "train")
SOURCE_PYTHONPATH_RELATIVES = (
    ".",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM",
)


class QualificationError(RuntimeError):
    """A fail-closed qualification contract violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    """Write canonical JSON exactly once, fsync it, and return file SHA256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(payload)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return hashlib.sha256(encoded).hexdigest()


def _real_directory(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise QualificationError(f"{field} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"{field} does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise QualificationError(f"{field} must be a directory: {resolved}")
    return resolved


def _real_file(path: str | Path, *, field: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise QualificationError(f"{field} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"{field} does not exist: {candidate}") from exc
    if not resolved.is_file():
        raise QualificationError(f"{field} must be a regular file: {resolved}")
    return resolved


def _under(path: Path, root: Path, *, field: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise QualificationError(f"{field} escaped {root}: {resolved}")
    return resolved


def _distribution_module_provenance(
    module: Any,
    distribution_name: str,
    *,
    source_root: Path,
) -> dict[str, str]:
    """Bind an imported top-level package to this interpreter's metadata.

    ``uv`` may install an individual package file as a symlink into an
    image-owned cache. Requiring the resolved module path to remain below the
    lexical ``site-packages`` directory would reject that valid layout. The
    stronger portable identity is that the imported file equals the exact
    top-level package anchor selected by ``importlib.metadata``.
    """
    from importlib import metadata

    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise QualificationError(
            f"runtime dependency {module.__name__!r} has no file identity"
        )
    module_path = Path(module_file).resolve(strict=True)
    top_level_name = module.__name__.split(".", 1)[0]
    distribution = metadata.distribution(distribution_name)
    anchor_relative = Path(top_level_name) / Path(module_file).name
    anchor_path = Path(distribution.locate_file(anchor_relative)).resolve(strict=True)
    if module_path != anchor_path:
        raise QualificationError(
            "runtime dependency does not match its interpreter-selected "
            f"distribution anchor: {module.__name__}={module_path}, "
            f"anchor={anchor_path}"
        )
    if module_path.is_relative_to(source_root):
        raise QualificationError(
            f"runtime dependency came from source mount: "
            f"{module.__name__}={module_path}"
        )
    return {
        "module_path": str(module_path),
        "distribution_anchor": str(anchor_path),
        "distribution_root": str(Path(distribution.locate_file("")).absolute()),
        "distribution_version": distribution.version,
    }


def validated_source_pythonpath(source_root: Path) -> str:
    """Return the complete, source-owned import path for Megatron actors.

    The official image's prefetched MCore environment contains the native
    dependencies, but its editable source links point at ``/opt/nemo-rl``.
    Qualification must instead import NeMo-RL, Megatron-Bridge, and
    Megatron-Core from the one sealed read-only source bundle.  A free-form
    caller ``PYTHONPATH`` is never inherited.
    """
    root = _real_directory(source_root, field="source_root")
    entries: list[str] = []
    for relative in SOURCE_PYTHONPATH_RELATIVES:
        candidate = root if relative == "." else root / relative
        entries.append(
            str(
                _under(
                    _real_directory(candidate, field="source_pythonpath"),
                    root,
                    field="source_pythonpath",
                )
            )
        )
    if len(set(entries)) != len(entries):
        raise QualificationError("source Python path contains duplicate roots")
    return os.pathsep.join(entries)


def create_fresh_run_root(output_parent: Path, run_id: str) -> Path:
    """Create an empty, attempt-scoped root; existing runs are never resumed."""
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise QualificationError(
            "run_id must be 8-128 characters from [A-Za-z0-9._-]"
        )
    parent = _real_directory(output_parent, field="output_parent")
    run_root = parent / run_id
    if run_root.exists() or run_root.is_symlink():
        raise QualificationError(
            f"qualification run root already exists; replay is forbidden: {run_root}"
        )
    run_root.mkdir(mode=0o750)
    for relative in (
        "cache",
        "checkpoint",
        "debug/actors",
        "debug/controller",
        "evidence",
        "journal",
        "r3",
    ):
        (run_root / relative).mkdir(mode=0o750, parents=True)
    return run_root.resolve(strict=True)


def record_stage_event(
    run_root: Path,
    *,
    run_id: str,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> str:
    """Append one immutable controller-stage event for post-mortem diagnosis."""
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", stage) is None:
        raise QualificationError(f"invalid diagnostic stage: {stage!r}")
    if status not in {"started", "completed", "failed"}:
        raise QualificationError(f"invalid diagnostic stage status: {status!r}")
    payload = {
        "format": "nemo-rl-production-nano-tq-stage-event-v1",
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "ray_address": os.environ.get("RAY_ADDRESS"),
        "time_ns": time.time_ns(),
        "details": dict(details or {}),
    }
    file_name = f"{payload['time_ns']}-{stage}-{status}.json"
    return write_json_exclusive(
        run_root / "debug" / "controller" / file_name,
        payload,
    )


def configure_attempt_cache(run_root: Path) -> dict[str, str]:
    """Pin every mutable Python/model cache below the fresh attempt root."""
    cache_root = _under(
        _real_directory(run_root / "cache", field="cache_root"),
        run_root,
        field="cache_root",
    )
    paths = {
        "HF_HOME": cache_root / "huggingface",
        "HF_MODULES_CACHE": cache_root / "huggingface" / "modules",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "NEMO_HOME": cache_root / "nemo",
        "NRL_MEGATRON_CHECKPOINT_DIR": cache_root / "megatron-checkpoint",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    normalized: dict[str, str] = {}
    for name, path in paths.items():
        path.mkdir(parents=True, mode=0o750, exist_ok=False)
        normalized[name] = str(path.resolve(strict=True))
        os.environ[name] = normalized[name]
    return normalized


def _mountinfo_record(path: Path, *, expected_read_only: bool) -> dict[str, Any]:
    """Return the longest matching Linux mount record and enforce its mode."""
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        raise QualificationError("/proc/self/mountinfo is unavailable")

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
            raise QualificationError("malformed /proc/self/mountinfo record")
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 6 or len(right_fields) < 3:
            raise QualificationError("truncated /proc/self/mountinfo record")
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
        raise QualificationError(f"no mountinfo authority covers {resolved}")
    mount_point, mount_options, super_options, fs_type, source = max(
        candidates, key=lambda item: len(item[0].parts)
    )
    observed_read_only = "ro" in mount_options and "rw" not in mount_options
    if observed_read_only != expected_read_only:
        raise QualificationError(
            f"mount mode changed for {resolved}: options={mount_options}, "
            f"expected_read_only={expected_read_only}"
        )
    return {
        "path": str(resolved),
        "mount_point": str(mount_point),
        "mount_options": mount_options,
        "super_options": super_options,
        "filesystem_type": fs_type,
        "source": source,
        "read_only": observed_read_only,
    }


def validate_static_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the dependency-free, controller-owned qualification contract."""
    normalized = json.loads(json.dumps(contract))
    required = {
        "run_id",
        "source_stack_id",
        "source_bundle_manifest_sha256",
        "model_manifest_sha256",
        "expected_image_digest",
        "expected_image_fingerprint_sha256",
        "expected_driver_venv",
        "expected_actor_venv",
        "expected_python_version",
        "expected_ray_version",
        "expected_torch_version",
        "expected_cuda_compute_capability",
        "expected_num_nodes",
        "expected_gpus_per_node",
        "expected_world_size",
        "expected_tensor_parallel_size",
        "expected_pipeline_parallel_size",
        "expected_context_parallel_size",
        "expected_expert_parallel_size",
        "expected_train_global_batch_size",
        "expected_train_micro_batch_size",
        "expected_sequence_packing_enabled",
        "expected_data_plane_backend",
        "checkpoint_required",
        "restart_safe_replay",
    }
    if set(normalized) != required:
        missing = sorted(required - set(normalized))
        extra = sorted(set(normalized) - required)
        raise QualificationError(
            f"static contract keys changed: missing={missing}, extra={extra}"
        )
    if RUN_ID_RE.fullmatch(normalized["run_id"]) is None:
        raise QualificationError("invalid static contract run_id")
    for key in ("source_stack_id",):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise QualificationError(f"{key} must be a non-empty string")
    for key in (
        "source_bundle_manifest_sha256",
        "model_manifest_sha256",
        "expected_image_fingerprint_sha256",
    ):
        if not isinstance(normalized[key], str) or SHA256_RE.fullmatch(
            normalized[key]
        ) is None:
            raise QualificationError(f"{key} must be SHA256 hex")
    if (
        not isinstance(normalized["expected_image_digest"], str)
        or IMAGE_DIGEST_RE.fullmatch(normalized["expected_image_digest"]) is None
    ):
        raise QualificationError("expected_image_digest must be a full sha256 digest")
    for key in (
        "expected_num_nodes",
        "expected_gpus_per_node",
        "expected_world_size",
        "expected_tensor_parallel_size",
        "expected_pipeline_parallel_size",
        "expected_context_parallel_size",
        "expected_expert_parallel_size",
        "expected_train_global_batch_size",
        "expected_train_micro_batch_size",
    ):
        if (
            isinstance(normalized[key], bool)
            or not isinstance(normalized[key], int)
            or normalized[key] <= 0
        ):
            raise QualificationError(f"{key} must be a positive integer")
    if normalized["expected_world_size"] != (
        normalized["expected_num_nodes"] * normalized["expected_gpus_per_node"]
    ):
        raise QualificationError(
            "expected_world_size must equal expected_num_nodes * "
            "expected_gpus_per_node"
        )
    if normalized["expected_context_parallel_size"] != 1:
        raise QualificationError(
            "this production fixed-batch harness is CP1-only; use the separate "
            "CP2 lane after CP1 qualifies"
        )
    if normalized["expected_sequence_packing_enabled"] is not False:
        raise QualificationError(
            "the fixed-batch qualification requires sequence packing disabled"
        )
    if not isinstance(
        normalized["expected_cuda_compute_capability"], str
    ) or re.fullmatch(
        r"[0-9]+\.[0-9]+", normalized["expected_cuda_compute_capability"]
    ) is None:
        raise QualificationError(
            "expected_cuda_compute_capability must be an exact major.minor value"
        )
    if normalized["expected_data_plane_backend"] not in {
        "simple",
        "mooncake_cpu",
    }:
        raise QualificationError("unsupported data-plane backend")
    if normalized["checkpoint_required"] is not True:
        raise QualificationError("production qualification requires a checkpoint")
    if normalized["restart_safe_replay"] is not False:
        raise QualificationError(
            "optimizer RPC ambiguity is not replay-safe in this harness"
        )
    for key in ("expected_driver_venv", "expected_actor_venv"):
        if not isinstance(normalized[key], str) or not normalized[key].startswith("/"):
            raise QualificationError(f"{key} must be absolute")
    for key in (
        "expected_python_version",
        "expected_ray_version",
        "expected_torch_version",
    ):
        if not isinstance(normalized[key], str) or re.fullmatch(
            r"[0-9]+\.[0-9]+\.[0-9]+", normalized[key]
        ) is None:
            raise QualificationError(f"{key} must be an exact X.Y.Z version")
    return normalized


def validate_config_projection(
    projection: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the resolved production semantics before any Ray actor starts."""
    value = json.loads(json.dumps(projection))
    if value.get("data_plane_enabled") is not True:
        raise QualificationError("data_plane.enabled must resolve to true")
    if value.get("data_plane_impl") != "transfer_queue":
        raise QualificationError("data_plane.impl must be transfer_queue")
    if value.get("data_plane_backend") != contract[
        "expected_data_plane_backend"
    ]:
        raise QualificationError("resolved data-plane backend changed")
    if value.get("policy_backend") != "megatron":
        raise QualificationError("production Nano qualification requires Megatron")
    if value.get("is_vlm") is not True:
        raise QualificationError("production Nano qualification requires is_vlm=true")
    if value.get("router_replay_enabled") is not False:
        raise QualificationError(
            "fixed-batch qualification has no rollout route authority; disable router replay"
        )
    if value.get("tokenizer_use_fastokens") is not False:
        raise QualificationError("fixed-batch qualification disables Fastokens")
    if value.get("generation_refit_transport") is not None:
        raise QualificationError(
            "fixed-batch qualification disables generation/refit transports"
        )
    if value.get("context_parallel_size") != contract[
        "expected_context_parallel_size"
    ]:
        raise QualificationError("resolved context parallel size changed")
    if value.get("world_size") != contract["expected_world_size"]:
        raise QualificationError("resolved worker world size changed")
    exact_projection_fields = {
        "num_nodes": "expected_num_nodes",
        "gpus_per_node": "expected_gpus_per_node",
        "tensor_parallel_size": "expected_tensor_parallel_size",
        "pipeline_parallel_size": "expected_pipeline_parallel_size",
        "expert_parallel_size": "expected_expert_parallel_size",
        "train_global_batch_size": "expected_train_global_batch_size",
        "train_micro_batch_size": "expected_train_micro_batch_size",
        "sequence_packing_enabled": "expected_sequence_packing_enabled",
    }
    for projection_key, contract_key in exact_projection_fields.items():
        if value.get(projection_key) != contract[contract_key]:
            raise QualificationError(
                f"resolved {projection_key} changed: "
                f"{value.get(projection_key)!r} != {contract[contract_key]!r}"
            )
    if value["train_global_batch_size"] <= 1:
        raise QualificationError("qualification needs at least two GRPO rows")
    if value["train_global_batch_size"] % value["data_parallel_size"]:
        raise QualificationError("global batch is not divisible by data parallel size")
    if value["data_parallel_size"] != 1:
        raise QualificationError(
            "the CP1 trace-join contract currently requires data_parallel_size=1"
        )
    if value.get("reference_policy_kl_penalty", 0.0) <= 0:
        raise QualificationError(
            "reference_policy_kl_penalty must be positive so ref_lp is real"
        )
    if value.get("dynamic_batching_enabled") is not False:
        raise QualificationError("fixed-batch qualification disables dynamic batching")
    if value.get("mtp_num_layers") not in (None, 0):
        raise QualificationError("Nano media qualification does not support MTP")
    if value.get("fused_linear_logprobs") is not False:
        raise QualificationError("Nano media qualification disables fused logprobs")
    if value.get("virtual_pipeline_size") not in (None, 1):
        raise QualificationError("Nano media qualification disables virtual PP")
    if value.get("train_micro_batch_size", 0) <= 0:
        raise QualificationError("train micro batch size must be positive")
    return value


def validate_media_trace_join(
    *,
    schema: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    sample_ids: Sequence[str],
    expected_world_size: int,
) -> dict[str, Any]:
    """Join every prev/ref/train worker fetch to one media wire authority."""
    wire_schema_id = schema.get("wire_schema_id")
    entries = schema.get("entries")
    if not isinstance(wire_schema_id, str) or SHA256_RE.fullmatch(
        wire_schema_id
    ) is None:
        raise QualificationError("invalid media wire schema ID")
    if not isinstance(entries, list) or not entries:
        raise QualificationError("media wire schema has no entries")
    authority = {
        (sample_id, entry["logical_key"]): entry[
            "row_sha256_by_sample_id"
        ][sample_id]
        for entry in entries
        for sample_id in sample_ids
    }
    observed: dict[tuple[str, str, str], set[int]] = {}
    for record in records:
        if record.get("event") != "tq_fetch_sample":
            continue
        stage = record.get("stage")
        sample_id = record.get("key")
        if stage not in REQUIRED_STAGES or sample_id not in sample_ids:
            continue
        if record.get("media_wire_schema_id") != wire_schema_id:
            raise QualificationError(
                f"worker media schema changed at {stage}/{sample_id}"
            )
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise QualificationError("R3 media record is missing a distributed rank")
        media = record.get("packed_tensor_media")
        if not isinstance(media, dict) or set(media) != {
            entry["logical_key"] for entry in entries
        }:
            raise QualificationError(
                f"worker media fields changed at {stage}/{sample_id}"
            )
        for logical_key, tensor_record in media.items():
            expected = authority[(sample_id, logical_key)]
            actual = (
                None
                if tensor_record is None
                else sha256_json(
                    {
                        "dtype": tensor_record.get("dtype"),
                        "shape": tensor_record.get("shape"),
                        "bytes_sha256": tensor_record.get("sha256"),
                    }
                )
            )
            if actual != expected:
                raise QualificationError(
                    f"media digest mismatch at {stage}/{sample_id}/{logical_key}"
                )
            observed.setdefault((stage, sample_id, logical_key), set()).add(rank)

    missing = []
    for stage in REQUIRED_STAGES:
        for sample_id in sample_ids:
            for entry in entries:
                key = (stage, sample_id, entry["logical_key"])
                ranks = observed.get(key, set())
                if ranks != set(range(expected_world_size)):
                    missing.append(
                        {
                            "stage": stage,
                            "sample_id": sample_id,
                            "logical_key": entry["logical_key"],
                            "ranks": sorted(ranks),
                        }
                    )
    if missing:
        raise QualificationError(
            "media trace join is incomplete: "
            + json.dumps(missing[:8], sort_keys=True)
        )
    return {
        "wire_schema_id": wire_schema_id,
        "stages": list(REQUIRED_STAGES),
        "sample_count": len(sample_ids),
        "logical_media_keys": sorted(
            entry["logical_key"] for entry in entries
        ),
        "rank_count": expected_world_size,
        "joined_record_keys": len(observed),
    }


def collect_r3_records(trace_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(trace_root.glob("r3_trace_*_pid*.jsonl")):
        if path.is_symlink() or not path.is_file():
            raise QualificationError(f"invalid R3 evidence path: {path}")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise QualificationError(
                    f"invalid R3 JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise QualificationError(f"non-object R3 record at {path}:{line_number}")
            records.append(value)
    if not records:
        raise QualificationError(f"no R3 records found under {trace_root}")
    return records


def checkpoint_tree_manifest(checkpoint_root: Path) -> dict[str, Any]:
    """Hash a completed, symlink-free checkpoint tree."""
    root = _real_directory(checkpoint_root, field="checkpoint_root")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise QualificationError(f"checkpoint contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QualificationError(f"checkpoint contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        files.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "mode": stat.st_mode & 0o777,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise QualificationError("checkpoint tree is empty")
    payload = {
        "format": CHECKPOINT_MANIFEST_FORMAT,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    payload["checkpoint_tree_sha256"] = sha256_json(payload)
    return payload


def verify_optimizer_journal(
    journal_root: Path,
    *,
    run_id: str,
    world_size: int,
    checkpoint_join_required: bool,
    expected_checkpoint_tree_sha256: str | None = None,
    expected_controller_result_sha256: str | None = None,
) -> dict[str, Any]:
    if checkpoint_join_required and (
        expected_checkpoint_tree_sha256 is None
    ) != (expected_controller_result_sha256 is None):
        raise QualificationError(
            "checkpoint and controller join authorities must be supplied together"
        )
    required_phases = ["baseline", "optimizer-dispatched", "optimizer-applied"]
    if checkpoint_join_required:
        required_phases.append("checkpoint-joined")
    rank_records: list[dict[str, Any]] = []
    for rank in range(world_size):
        rank_root = journal_root / f"rank-{rank:05d}"
        for forbidden in (
            "optimizer-outcome-ambiguous",
            "optimizer-applied-without-parameter-delta",
            "optimizer-applied-with-invalid-gradient-evidence",
        ):
            if (rank_root / f"{forbidden}.json").exists():
                raise QualificationError(
                    f"rank {rank} journal contains terminal failure {forbidden}"
                )
        phases: dict[str, dict[str, Any]] = {}
        for phase in required_phases:
            path = _real_file(rank_root / f"{phase}.json", field=phase)
            raw = path.read_bytes()
            payload = json.loads(raw)
            if raw != canonical_json_bytes(payload) + b"\n":
                raise QualificationError(f"journal is not canonical JSON: {path}")
            if (
                payload.get("format")
                != "nemo-rl-production-nano-tq-optimizer-journal-v1"
                or payload.get("phase") != phase
                or payload.get("run_id") != run_id
                or payload.get("rank") != rank
            ):
                raise QualificationError(f"journal identity mismatch in {path}")
            phases[phase] = payload
        identity_fields = (
            "step_id",
            "partition_id",
            "sample_ids",
            "media_wire_schema_id",
        )
        for field in identity_fields:
            if any(
                phase[field] != phases["baseline"][field]
                for phase in phases.values()
            ):
                raise QualificationError(
                    f"rank {rank} journal changed {field} across phases"
                )
        before = phases["baseline"]["parameter_state"]["parameter_sha256"]
        applied = phases["optimizer-applied"]
        after = applied["post_parameter_state"]["parameter_sha256"]
        gradient_sha256 = applied["post_parameter_state"]["gradient_sha256"]
        if any(SHA256_RE.fullmatch(value) is None for value in (before, after)):
            raise QualificationError(f"rank {rank} parameter digest is invalid")
        if SHA256_RE.fullmatch(gradient_sha256) is None:
            raise QualificationError(f"rank {rank} gradient digest is invalid")
        if before == after or applied.get("parameter_delta") is not True:
            raise QualificationError(f"rank {rank} has no parameter delta")
        if (
            applied["post_parameter_state"].get("gradients_finite") is not True
            or applied["post_parameter_state"].get("gradients_nonzero") is not True
        ):
            raise QualificationError(
                f"rank {rank} has no finite nonzero gradient evidence"
            )
        if applied.get("restart_safe_replay") is not False:
            raise QualificationError("journal incorrectly claims replay safety")
        if checkpoint_join_required:
            joined = phases["checkpoint-joined"]
            if expected_checkpoint_tree_sha256 is not None and joined.get(
                "checkpoint_tree_sha256"
            ) != expected_checkpoint_tree_sha256:
                raise QualificationError(
                    f"rank {rank} checkpoint tree join changed"
                )
            if expected_controller_result_sha256 is not None and joined.get(
                "controller_result_sha256"
            ) != expected_controller_result_sha256:
                raise QualificationError(
                    f"rank {rank} controller result join changed"
                )
        rank_records.append(
            {
                "rank": rank,
                "before_parameter_sha256": before,
                "after_parameter_sha256": after,
                "gradient_sha256": gradient_sha256,
                "phase_file_sha256": {
                    phase: sha256_file(rank_root / f"{phase}.json")
                    for phase in required_phases
                },
            }
        )
    return {
        "format": "nemo-rl-production-nano-tq-optimizer-journal-summary-v1",
        "run_id": run_id,
        "world_size": world_size,
        "required_phases": required_phases,
        "rank_records": rank_records,
        "restart_safe_replay": False,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "numel"):
        tensor = value.detach().to("cpu")
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _tensor_record(tensor: Any) -> dict[str, Any]:
    value = tensor.detach().to("cpu").contiguous()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(
            value.view(__import__("torch").uint8).numpy().tobytes()
        ).hexdigest(),
    }


def _build_fixed_batch(processor: Any, tokenizer: Any, *, batch_size: int, config: Any):
    """Build deterministic, non-cancelling image/completion rows."""
    import torch
    from PIL import Image, ImageDraw

    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
    )
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.data.processors import vlm_hf_data_processor
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    completion_texts = (
        "The green square is on the left.",
        "The blue circle is on the right.",
        "The upper object is brighter.",
        "The lower object is darker.",
    )
    message_logs = []
    for row in range(batch_size):
        width = 96 if row % 2 == 0 else 64
        height = 64 if row % 2 == 0 else 96
        image = Image.new("RGB", (width, height), color=(13, 29, 47))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (4 + row % 7, 5, min(width - 5, 38), min(height - 5, 42)),
            fill=(20 + row * 7 % 200, 190, 60),
        )
        draw.ellipse(
            (max(1, width - 34), max(1, height - 34), width - 3, height - 3),
            fill=(30, 80 + row * 11 % 160, 210),
        )
        datum = {
            "task_name": "daily-omni",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": "Describe one visible spatial relation in one sentence.",
                        },
                    ],
                },
                {"role": "assistant", "content": completion_texts[row % 4]},
            ],
        }
        processed = vlm_hf_data_processor(
            datum,
            TaskDataSpec(task_name="daily-omni"),
            processor,
            int(config.policy["max_total_sequence_length"]),
            row,
        )
        message_log = list(processed["message_log"])
        completion = completion_texts[row % 4]
        completion_ids = tokenizer(
            completion,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        if tokenizer.eos_token_id is not None:
            completion_ids = torch.cat(
                [completion_ids, torch.tensor([tokenizer.eos_token_id])]
            )
        message_log.append(
            {
                "role": "assistant",
                "content": completion,
                "token_ids": completion_ids.to(torch.long),
                # Presence marks these tokens as rollout-generated.  The values
                # are intentionally not seeded into TQ; after prev_lp returns,
                # that exact tensor is written as generation_logprobs.
                "generation_logprobs": torch.zeros(
                    len(completion_ids), dtype=torch.float32
                ),
            }
        )
        message_logs.append(message_log)

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": int(tokenizer.pad_token_id or 0)},
        make_sequence_length_divisible_by=int(
            config.policy["make_sequence_length_divisible_by"]
        ),
    )
    token_mask = flat["token_loss_mask"].to(torch.float32)
    if token_mask.shape != flat["token_ids"].shape:
        raise QualificationError("fixed batch token mask shape changed")
    action_counts = token_mask.sum(dim=1)
    if not bool(torch.all(action_counts > 0).item()):
        raise QualificationError("every fixed-batch row must contain action tokens")

    batch = BatchedDataDict(
        {
            "input_ids": flat["token_ids"].to(torch.long),
            "input_lengths": input_lengths.to(torch.int32),
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size, dtype=torch.float32),
        }
    )
    batch.update(flat.get_multimodal_dict(as_tensors=False))
    media_keys = sorted(
        key for key, value in batch.items() if value.__class__.__name__ == "PackedTensor"
    )
    if "pixel_values" not in media_keys or "imgs_sizes" not in media_keys:
        raise QualificationError(
            f"Nano processor did not produce pixel_values+imgs_sizes: {media_keys}"
        )
    batch.to("cpu")
    return batch, {
        "completion_texts": [completion_texts[row % 4] for row in range(batch_size)],
        "input_ids": _tensor_record(batch["input_ids"]),
        "input_lengths": _tensor_record(batch["input_lengths"]),
        "token_mask": _tensor_record(batch["token_mask"]),
        "action_token_counts": [int(value) for value in action_counts.tolist()],
        "media_keys": media_keys,
    }


def _build_config_projection(config: Any) -> dict[str, Any]:
    policy = config.policy
    megatron = policy["megatron_cfg"]
    tp = int(megatron["tensor_model_parallel_size"])
    pp = int(megatron.get("pipeline_model_parallel_size", 1))
    cp = int(megatron.get("context_parallel_size", 1))
    ep = int(megatron.get("expert_model_parallel_size", 1))
    world_size = int(config.cluster["num_nodes"]) * int(
        config.cluster["gpus_per_node"]
    )
    model_parallel = tp * pp * cp
    if world_size % model_parallel:
        raise QualificationError("world size is not divisible by TP*PP*CP")
    return {
        "data_plane_enabled": bool(config.data_plane["enabled"]),
        "data_plane_impl": config.data_plane["impl"],
        "data_plane_backend": config.data_plane["backend"],
        "policy_backend": "megatron"
        if policy.get("megatron_cfg", {}).get("enabled", True)
        else "other",
        "model_name": policy["model_name"],
        "is_vlm": bool(policy.get("is_vlm")),
        "num_nodes": int(config.cluster["num_nodes"]),
        "gpus_per_node": int(config.cluster["gpus_per_node"]),
        "world_size": world_size,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "context_parallel_size": cp,
        "expert_parallel_size": ep,
        "data_parallel_size": world_size // model_parallel,
        "train_global_batch_size": int(policy["train_global_batch_size"]),
        "train_micro_batch_size": int(policy["train_micro_batch_size"]),
        "sequence_packing_enabled": bool(policy["sequence_packing"]["enabled"]),
        "dynamic_batching_enabled": bool(policy["dynamic_batching"]["enabled"]),
        "router_replay_enabled": bool(
            (policy.get("router_replay") or {}).get("enabled", False)
        ),
        "tokenizer_use_fastokens": bool(
            (policy.get("tokenizer") or {}).get("use_fastokens", False)
        ),
        "generation_refit_transport": (policy.get("generation") or {}).get(
            "refit_transport"
        ),
        "reference_policy_kl_penalty": float(
            config.loss_fn.reference_policy_kl_penalty
        ),
        "mtp_num_layers": megatron.get("mtp_num_layers"),
        "fused_linear_logprobs": bool(
            megatron.get("use_fused_linear_logprobs", False)
        ),
        "virtual_pipeline_size": megatron.get(
            "virtual_pipeline_model_parallel_size"
        ),
    }


def _validate_driver_provenance(
    *,
    source_root: Path,
    driver_venv: Path,
    actor_venv: Path,
    model_root: Path,
    run_root: Path,
    expected_image_digest: str,
    expected_image_fingerprint_sha256: str,
    expected_python_version: str,
    expected_ray_version: str,
    expected_torch_version: str,
) -> dict[str, Any]:
    import mooncake
    import mooncake.store
    import nemo_rl
    import packaging
    import ray
    import tensordict
    import torch
    import transfer_queue
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    from nemo_rl.data_plane.adapters.transfer_queue import (
        validate_baked_transfer_queue,
    )

    if Path(sys.prefix).resolve(strict=True) != driver_venv:
        raise QualificationError(
            f"wrong driver venv: prefix={sys.prefix}, expected={driver_venv}"
        )
    if (
        not sys.dont_write_bytecode
        or not sys.flags.no_user_site
        or not sys.flags.safe_path
    ):
        raise QualificationError(
            "driver must use -B -P -s (no bytecode, cwd path, or user site)"
        )
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    if python_version != expected_python_version:
        raise QualificationError(
            f"driver Python changed: {python_version} != {expected_python_version}"
        )
    if ray.__version__ != expected_ray_version:
        raise QualificationError(
            f"driver Ray changed: {ray.__version__} != {expected_ray_version}"
        )
    torch_release = torch.__version__.split("+", 1)[0]
    if torch_release != expected_torch_version:
        raise QualificationError(
            f"driver Torch changed: {torch.__version__} != {expected_torch_version}"
        )
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise QualificationError("production qualification requires Slurm job+step IDs")
    observed_image_digest = os.environ.get("NRL_QUALIFIED_IMAGE_DIGEST")
    if observed_image_digest != expected_image_digest:
        raise QualificationError(
            "scheduler image digest changed: "
            f"{observed_image_digest!r} != {expected_image_digest!r}"
        )
    image_fingerprint_path = _real_file(
        "/opt/nemo_rl_container_fingerprint",
        field="image_fingerprint",
    )
    observed_fingerprint_sha256 = sha256_file(image_fingerprint_path)
    if observed_fingerprint_sha256 != expected_image_fingerprint_sha256:
        raise QualificationError(
            "image fingerprint changed: "
            f"{observed_fingerprint_sha256} != "
            f"{expected_image_fingerprint_sha256}"
        )

    import tomllib

    pyproject_path = source_root / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text())
    python_specifier = pyproject["project"]["requires-python"]
    if Version(python_version) not in SpecifierSet(python_specifier):
        raise QualificationError(
            "runtime Python is outside mounted source requires-python: "
            f"runtime={python_version}, requires-python={python_specifier}"
        )
    expected_nemo_rl_path = source_root / "nemo_rl" / "__init__.py"
    nemo_rl_path = _under(
        Path(nemo_rl.__file__), source_root, field="module nemo_rl"
    )
    if nemo_rl_path != expected_nemo_rl_path.resolve(strict=True):
        raise QualificationError(
            f"nemo_rl imported from the wrong source path: {nemo_rl_path}"
        )
    source_paths = {"nemo_rl": str(nemo_rl_path)}
    dependency_paths: dict[str, dict[str, str]] = {}
    for module, distribution_name in (
        (torch, "torch"),
        (ray, "ray"),
        (transfer_queue, "TransferQueue"),
        (tensordict, "tensordict"),
        (mooncake, "mooncake-transfer-engine-cuda13"),
        (packaging, "packaging"),
    ):
        dependency_paths[module.__name__] = _distribution_module_provenance(
            module,
            distribution_name,
            source_root=source_root,
        )
    transfer_queue_provenance = validate_baked_transfer_queue()
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "python_flags": {
            "dont_write_bytecode": bool(sys.dont_write_bytecode),
            "no_user_site": bool(sys.flags.no_user_site),
            "safe_path": bool(sys.flags.safe_path),
        },
        "source_root": str(source_root),
        "source_modules": source_paths,
        "source_pythonpath": validated_source_pythonpath(source_root),
        "source_python_requires": python_specifier,
        "driver_venv": str(driver_venv),
        "actor_venv": str(actor_venv),
        "model_root": str(model_root),
        "run_root": str(run_root),
        "image_digest": expected_image_digest,
        "image_fingerprint": {
            "path": str(image_fingerprint_path),
            "sha256": observed_fingerprint_sha256,
        },
        "mounts": {
            "source": _mountinfo_record(source_root, expected_read_only=True),
            "model": _mountinfo_record(model_root, expected_read_only=True),
            "output": _mountinfo_record(run_root, expected_read_only=False),
        },
        "scheduler": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "step_id": os.environ["SLURM_STEP_ID"],
        },
        "mooncake_store_module": str(
            Path(mooncake.store.__file__).resolve(strict=True)
        ),
        "python_version": python_version,
        "ray_version": ray.__version__,
        "torch_version": torch.__version__,
        "dependency_modules": dependency_paths,
        "transfer_queue": transfer_queue_provenance,
    }


def _load_resolved_config(config_path: Path, overrides: Sequence[str]) -> Any:
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = load_config(str(config_path))
    if overrides:
        config = parse_hydra_overrides(config, list(overrides))
    return MasterConfig(**OmegaConf.to_container(config, resolve=True))


def _load_processor(config: Any) -> tuple[Any, Any]:
    from nemo_rl.algorithms.utils import get_tokenizer

    processor = get_tokenizer(config.policy["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer
    return processor, tokenizer


def _scan_actor_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    run_id: str,
) -> list[dict[str, Any]]:
    if len(evidence) != world_size:
        raise QualificationError(
            f"actor evidence count {len(evidence)} != world size {world_size}"
        )
    by_rank = {}
    process_identities: set[tuple[str, int]] = set()
    for record in evidence:
        rank = record.get("rank")
        if (
            record.get("run_id") != run_id
            or record.get("world_size") != world_size
            or rank in by_rank
        ):
            raise QualificationError("actor evidence identity changed")
        process_identity = (record.get("hostname"), record.get("pid"))
        if (
            not isinstance(process_identity[0], str)
            or not process_identity[0]
            or isinstance(process_identity[1], bool)
            or not isinstance(process_identity[1], int)
            or process_identity in process_identities
        ):
            raise QualificationError("actor process identity is invalid or duplicated")
        process_identities.add(process_identity)
        by_rank[rank] = dict(record)
    if set(by_rank) != set(range(world_size)):
        raise QualificationError("actor evidence ranks are incomplete")
    return [by_rank[rank] for rank in range(world_size)]


def _run_production(args: argparse.Namespace, overrides: Sequence[str]) -> int:
    if os.environ.get("NRL_USE_FASTOKENS") not in (None, "0"):
        raise QualificationError(
            "NRL_USE_FASTOKENS must be absent or 0 for fixed-batch qualification"
        )
    source_root = _real_directory(args.expected_source_root, field="source_root")
    driver_venv = _real_directory(args.expected_driver_venv, field="driver_venv")
    actor_venv = _real_directory(args.expected_actor_venv, field="actor_venv")
    model_root = _real_directory(args.expected_model_root, field="model_root")
    config_path = _under(
        _real_file(args.config, field="config"), source_root, field="config"
    )
    source_manifest = _under(
        _real_file(args.source_bundle_manifest, field="source_bundle_manifest"),
        source_root,
        field="source_bundle_manifest",
    )
    model_manifest = _under(
        _real_file(args.model_manifest, field="model_manifest"),
        model_root,
        field="model_manifest",
    )
    source_manifest_sha = sha256_file(source_manifest)
    model_manifest_sha = sha256_file(model_manifest)
    if source_manifest_sha != args.source_bundle_manifest_sha256:
        raise QualificationError("source bundle manifest SHA256 changed")
    if model_manifest_sha != args.model_manifest_sha256:
        raise QualificationError("model manifest SHA256 changed")

    contract = validate_static_contract(
        {
            "run_id": args.run_id,
            "source_stack_id": args.source_stack_id,
            "source_bundle_manifest_sha256": source_manifest_sha,
            "model_manifest_sha256": model_manifest_sha,
            "expected_image_digest": args.expected_image_digest,
            "expected_image_fingerprint_sha256": (
                args.expected_image_fingerprint_sha256
            ),
            "expected_driver_venv": str(driver_venv),
            "expected_actor_venv": str(actor_venv),
            "expected_python_version": args.expected_python_version,
            "expected_ray_version": args.expected_ray_version,
            "expected_torch_version": args.expected_torch_version,
            "expected_cuda_compute_capability": (
                args.expected_cuda_compute_capability
            ),
            "expected_num_nodes": args.expected_num_nodes,
            "expected_gpus_per_node": args.expected_gpus_per_node,
            "expected_world_size": args.expected_world_size,
            "expected_tensor_parallel_size": args.expected_tensor_parallel_size,
            "expected_pipeline_parallel_size": args.expected_pipeline_parallel_size,
            "expected_context_parallel_size": args.expected_context_parallel_size,
            "expected_expert_parallel_size": args.expected_expert_parallel_size,
            "expected_train_global_batch_size": args.expected_train_global_batch_size,
            "expected_train_micro_batch_size": args.expected_train_micro_batch_size,
            "expected_sequence_packing_enabled": False,
            "expected_data_plane_backend": args.expected_data_plane_backend,
            "checkpoint_required": True,
            "restart_safe_replay": False,
        }
    )
    run_root = create_fresh_run_root(
        Path(args.output_parent),
        args.run_id,
    )
    attempt_cache = configure_attempt_cache(run_root)
    intent = {
        "format": RUN_INTENT_FORMAT,
        "contract": contract,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "hydra_overrides": list(overrides),
        "model_root": str(model_root),
        "source_root": str(source_root),
        "attempt_cache": attempt_cache,
        "time_ns": time.time_ns(),
    }
    intent_sha = write_json_exclusive(run_root / "RUN_INTENT.json", intent)

    policy = None
    meta = None
    optimizer_dispatched = False
    checkpoint_completed = False
    current_stage = "run-intent"

    def start_stage(stage: str, details: Mapping[str, Any] | None = None) -> None:
        nonlocal current_stage
        current_stage = stage
        record_stage_event(
            run_root,
            run_id=args.run_id,
            stage=stage,
            status="started",
            details=details,
        )

    def complete_stage(
        stage: str, details: Mapping[str, Any] | None = None
    ) -> None:
        if current_stage != stage:
            raise QualificationError(
                f"diagnostic stage changed: current={current_stage}, complete={stage}"
            )
        record_stage_event(
            run_root,
            run_id=args.run_id,
            stage=stage,
            status="completed",
            details=details,
        )

    try:
        start_stage("config")
        config = _load_resolved_config(config_path, overrides)
        projection = validate_config_projection(
            _build_config_projection(config), contract
        )
        projection_sha = write_json_exclusive(
            run_root / "evidence" / "RESOLVED_CONFIG_PROJECTION.json",
            {
                "format": "nemo-rl-production-nano-tq-config-projection-v1",
                "projection": projection,
            },
        )
        if Path(projection["model_name"]).resolve(strict=True) != model_root:
            raise QualificationError("resolved policy.model_name changed model root")
        complete_stage(
            "config",
            {
                "projection_sha256": projection_sha,
                "data_plane_backend": projection["data_plane_backend"],
                "world_size": projection["world_size"],
            },
        )

        # Qualification trace paths and source authority are scheduler-owned;
        # they are not inherited from a free-form caller environment.
        source_pythonpath = validated_source_pythonpath(source_root)
        worker_env = config.policy["megatron_cfg"].setdefault("env_vars", {})
        worker_env.update(
            {
                "B06_SOURCE_ROOT": str(source_root),
                "B06_EXPECTED_ACTOR_VENV": str(actor_venv),
                "NRL_QUALIFICATION_DEBUG_ROOT": str(run_root / "debug" / "actors"),
                "NRL_QUALIFIED_IMAGE_DIGEST": contract["expected_image_digest"],
                "NRL_R3_TRACE": "1",
                "NRL_R3_TRACE_STEPS": "1",
                "NRL_R3_TRACE_SAMPLES": str(projection["train_global_batch_size"]),
                "NRL_R3_TRACE_DIR": str(run_root / "r3"),
                "PYTHONPATH": source_pythonpath,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONSAFEPATH": "1",
                **attempt_cache,
            }
        )
        start_stage("driver-provenance")
        driver_provenance = _validate_driver_provenance(
            source_root=source_root,
            driver_venv=driver_venv,
            actor_venv=actor_venv,
            model_root=model_root,
            run_root=run_root,
            expected_image_digest=contract["expected_image_digest"],
            expected_image_fingerprint_sha256=contract[
                "expected_image_fingerprint_sha256"
            ],
            expected_python_version=contract["expected_python_version"],
            expected_ray_version=contract["expected_ray_version"],
            expected_torch_version=contract["expected_torch_version"],
        )
        complete_stage(
            "driver-provenance",
            {
                "image_digest": contract["expected_image_digest"],
                "source_manifest_sha256": source_manifest_sha,
                "model_manifest_sha256": model_manifest_sha,
            },
        )

        start_stage("fixed-batch")
        processor, tokenizer = _load_processor(config)
        batch, batch_evidence = _build_fixed_batch(
            processor,
            tokenizer,
            batch_size=projection["train_global_batch_size"],
            config=config,
        )
        sample_ids = [
            f"{args.run_id}:sample:{row:04d}"
            for row in range(projection["train_global_batch_size"])
        ]

        from nemo_rl.data_plane.packed_tensor_wire import (
            describe_packed_tensor_wire,
        )

        media_schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
        if media_schema is None:
            raise QualificationError("fixed Nano batch has no media wire schema")
        batch_evidence.update(
            {
                "sample_ids": sample_ids,
                "media_wire_schema": media_schema,
                "batch_id": sha256_json(
                    {
                        "sample_ids": sample_ids,
                        "input_ids": batch_evidence["input_ids"],
                        "input_lengths": batch_evidence["input_lengths"],
                        "token_mask": batch_evidence["token_mask"],
                        "media_wire_schema_id": media_schema["wire_schema_id"],
                    }
                ),
            }
        )
        batch_evidence_sha = write_json_exclusive(
            run_root / "evidence" / "FIXED_BATCH.json",
            {
                "format": "nemo-rl-production-nano-tq-fixed-batch-evidence-v1",
                **batch_evidence,
            },
        )
        complete_stage(
            "fixed-batch",
            {
                "batch_evidence_sha256": batch_evidence_sha,
                "batch_id": batch_evidence["batch_id"],
                "sample_count": len(sample_ids),
                "media_wire_schema_id": media_schema["wire_schema_id"],
            },
        )

        from nemo_rl.algorithms.loss import ClippedPGLossFn
        from nemo_rl.algorithms.utils import set_seed
        from nemo_rl.data_plane.column_io import kv_first_write
        from nemo_rl.data_plane.schema import PACKED_TENSOR_WIRE_SCHEMA_KEY
        from nemo_rl.distributed.ray_actor_environment_registry import (
            ACTOR_ENVIRONMENT_REGISTRY,
        )
        from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
        from nemo_rl.models.generation import configure_generation_config
        from nemo_rl.models.policy.tq_policy import TQPolicy

        start_stage("ray-policy-init")
        set_seed(int(config.grpo.seed))
        init_ray()
        training_resource = config.cluster.get("training_node_resource")
        constraints = (
            [{str(training_resource): 0.001}] * int(config.cluster["num_nodes"])
            if training_resource
            else None
        )
        cluster = RayVirtualCluster(
            name=f"nano_tq_qualification_{args.run_id}",
            bundle_ct_per_node_list=[int(config.cluster["gpus_per_node"])]
            * int(config.cluster["num_nodes"]),
            use_gpus=True,
            num_gpus_per_node=int(config.cluster["gpus_per_node"]),
            max_colocated_worker_groups=1,
            port_range_low=config.cluster.get("master_port_range_low"),
            port_range_high=config.cluster.get("master_port_range_high"),
            node_resource_constraints=constraints,
        )
        ACTOR_ENVIRONMENT_REGISTRY[WORKER_EXTENSION_FQN] = str(
            actor_venv / "bin" / "python"
        )
        config.policy["generation"] = configure_generation_config(
            config.policy["generation"], tokenizer
        )
        config.policy["generation"]["model_name"] = config.policy["model_name"]
        policy = TQPolicy(
            cluster=cluster,
            config=config.policy,
            tokenizer=tokenizer,
            processor=processor,
            weights_path=None,
            optimizer_path=None,
            init_optimizer=True,
            init_reference_model=True,
            worker_extension_cls_fqn=WORKER_EXTENSION_FQN,
            dp_cfg=config.data_plane,
            tq_partition_id=f"qualification-{args.run_id}",
        )
        complete_stage(
            "ray-policy-init",
            {
                "ray_address": os.environ.get("RAY_ADDRESS"),
                "ray_namespace": os.environ.get("RAY_NAMESPACE"),
                "partition_id": policy.tq_partition_id,
            },
        )

        start_stage("actor-provenance")
        actor_debug_root = str(run_root / "debug" / "actors")
        policy.run_all_workers_single_data(
            "qualification_debug_event",
            run_id=args.run_id,
            debug_root=actor_debug_root,
            stage="actor-provenance",
            status="started",
            details={"expected_world_size": projection["world_size"]},
        )
        actor_provenance = _scan_actor_evidence(
            policy.run_all_workers_single_data(
                "qualification_provenance",
                run_id=args.run_id,
                expected_source_root=str(source_root),
                expected_model_root=str(model_root),
                expected_run_root=str(run_root),
                expected_actor_venv=str(actor_venv),
                expected_image_digest=contract["expected_image_digest"],
                expected_python_version=contract["expected_python_version"],
                expected_ray_version=contract["expected_ray_version"],
                expected_torch_version=contract["expected_torch_version"],
                expected_cuda_compute_capability=contract[
                    "expected_cuda_compute_capability"
                ],
            ),
            world_size=projection["world_size"],
            run_id=args.run_id,
        )
        policy.run_all_workers_single_data(
            "qualification_debug_event",
            run_id=args.run_id,
            debug_root=actor_debug_root,
            stage="actor-provenance",
            status="completed",
            details={"actor_count": len(actor_provenance)},
        )
        actor_provenance_sha = write_json_exclusive(
            run_root / "evidence" / "ACTOR_PROVENANCE.json",
            {
                "format": "nemo-rl-production-nano-tq-actor-set-v1",
                "actors": actor_provenance,
            },
        )
        complete_stage(
            "actor-provenance",
            {"actor_provenance_sha256": actor_provenance_sha},
        )

        start_stage("tq-first-write")
        policy.prepare_step(
            num_samples=len(sample_ids),
            group_size=None,
            packed_tensor_wire_schema=media_schema,
        )
        tags = [
            {
                "qualification_run_id": args.run_id,
                "row_index": row,
                "batch_id": batch_evidence["batch_id"],
            }
            for row in range(len(sample_ids))
        ]
        meta = kv_first_write(
            batch,
            sample_ids=sample_ids,
            dp_client=policy.dp_client,
            partition_id=policy.tq_partition_id,
            task_name="train",
            pad_to_multiple=int(config.policy["make_sequence_length_divisible_by"]),
            tags=tags,
            extra_info={
                PACKED_TENSOR_WIRE_SCHEMA_KEY: media_schema,
                "qualification_run_id": args.run_id,
                "fixed_batch_id": batch_evidence["batch_id"],
            },
        )
        if (
            list(meta.sample_ids) != sample_ids
            or meta.partition_id != policy.tq_partition_id
            or (meta.extra_info or {}).get(PACKED_TENSOR_WIRE_SCHEMA_KEY)
            != media_schema
        ):
            raise QualificationError(
                "TQ first-write metadata changed fixed-batch identity"
            )
        complete_stage(
            "tq-first-write",
            {
                "partition_id": meta.partition_id,
                "sample_count": len(meta.sample_ids),
                "sample_ids_sha256": sha256_json(list(meta.sample_ids)),
                "media_wire_schema_id": media_schema["wire_schema_id"],
            },
        )

        start_stage("prev-lp")
        policy.prepare_for_lp_inference()
        policy.get_logprobs_from_meta(
            meta, micro_batch_size=int(config.policy["logprob_batch_size"])
        )
        prev = policy.read_from_dataplane(meta, select_fields=["prev_logprobs"])[
            "prev_logprobs"
        ]
        import torch

        effective_mask = batch["token_mask"].bool() & batch[
            "sample_mask"
        ].bool().unsqueeze(-1)
        if prev.shape != batch["token_mask"].shape:
            raise QualificationError("prev logprob shape changed")
        if not torch.isfinite(prev[effective_mask]).all():
            raise QualificationError("prev logprobs are non-finite on action tokens")
        if torch.count_nonzero(prev[~effective_mask]).item():
            raise QualificationError("prev logprobs are nonzero outside action tokens")
        complete_stage(
            "prev-lp",
            {
                "prev_logprobs_sha256": _tensor_record(prev)["sha256"],
                "eligible_action_tokens": int(effective_mask.sum().item()),
            },
        )

        centered = torch.arange(len(sample_ids), dtype=torch.float32)
        centered -= centered.mean()
        if torch.count_nonzero(centered).item() == 0:
            raise QualificationError("fixed advantages are degenerate")
        advantages = centered.unsqueeze(1).expand_as(prev) * batch["token_mask"]
        policy.write_to_dataplane(
            meta,
            {
                "generation_logprobs": prev.clone(),
                "advantages": advantages,
            },
        )
        start_stage("ref-lp")
        policy.get_reference_policy_logprobs_from_meta(
            meta, micro_batch_size=int(config.policy["logprob_batch_size"])
        )
        ref = policy.read_from_dataplane(
            meta, select_fields=["reference_policy_logprobs"]
        )["reference_policy_logprobs"]
        if ref.shape != prev.shape or not torch.isfinite(ref[effective_mask]).all():
            raise QualificationError("reference logprobs are invalid")
        if torch.count_nonzero(ref[~effective_mask]).item():
            raise QualificationError("reference logprobs are nonzero outside action tokens")
        complete_stage(
            "ref-lp",
            {
                "reference_logprobs_sha256": _tensor_record(ref)["sha256"],
            },
        )
        logprob_evidence = {
            "prev_logprobs": _tensor_record(prev),
            "reference_policy_logprobs": _tensor_record(ref),
            "generation_equals_prev": True,
            "eligible_action_tokens": int(effective_mask.sum().item()),
            "reference_max_abs_delta_from_prev": float(
                (ref - prev).abs()[effective_mask].max().item()
            ),
        }

        start_stage("optimizer-train")
        policy.prepare_for_training()
        step_id = f"{args.run_id}:optimizer-step:000000"
        baseline = _scan_actor_evidence(
            policy.run_all_workers_single_data(
                "qualification_arm_optimizer_step",
                run_id=args.run_id,
                step_id=step_id,
                journal_root=str(run_root / "journal"),
                partition_id=meta.partition_id,
                sample_ids=sample_ids,
                media_wire_schema_id=media_schema["wire_schema_id"],
            ),
            world_size=projection["world_size"],
            run_id=args.run_id,
        )
        optimizer_dispatched = True
        loss_fn = ClippedPGLossFn(
            config.loss_fn,
            use_fused_linear_logprobs=False,
        )
        train_result = policy.train_from_meta(
            meta,
            loss_fn,
            gbs=len(sample_ids),
            mbs=int(config.policy["train_micro_batch_size"]),
            scheduler_step_increment=len(sample_ids),
        )
        if not isinstance(train_result, Mapping) or not {
            "loss",
            "grad_norm",
        }.issubset(train_result):
            raise QualificationError(
                "training result is missing loss or grad_norm evidence"
            )
        loss_value = torch.as_tensor(train_result["loss"], dtype=torch.float64)
        grad_norm_value = torch.as_tensor(
            train_result["grad_norm"], dtype=torch.float64
        )
        if loss_value.numel() != 1 or not torch.isfinite(loss_value).all():
            raise QualificationError("training result has no finite scalar loss")
        if (
            grad_norm_value.numel() != 1
            or not torch.isfinite(grad_norm_value).all()
            or grad_norm_value.item() <= 0
        ):
            raise QualificationError("training result has no finite positive grad_norm")
        optimizer_evidence = _scan_actor_evidence(
            policy.run_all_workers_single_data(
                "qualification_optimizer_evidence", run_id=args.run_id
            ),
            world_size=projection["world_size"],
            run_id=args.run_id,
        )
        if not all(
            record["parameter_delta"]
            and record["gradients_finite"]
            and record["gradients_nonzero"]
            for record in optimizer_evidence
        ):
            raise QualificationError("optimizer rank evidence is incomplete")
        complete_stage(
            "optimizer-train",
            {
                "step_id": step_id,
                "loss": float(loss_value.item()),
                "grad_norm": float(grad_norm_value.item()),
                "optimizer_update_count": 1,
            },
        )

        start_stage("media-join")
        r3_records = collect_r3_records(run_root / "r3")
        media_join = validate_media_trace_join(
            schema=media_schema,
            records=r3_records,
            sample_ids=sample_ids,
            expected_world_size=projection["world_size"],
        )
        complete_stage(
            "media-join",
            {
                "wire_schema_id": media_join["wire_schema_id"],
                "joined_record_keys": media_join["joined_record_keys"],
            },
        )
        update_result = {
            "format": "nemo-rl-production-nano-tq-update-result-v1",
            "run_id": args.run_id,
            "step_id": step_id,
            "intent_sha256": intent_sha,
            "config_projection_sha256": projection_sha,
            "fixed_batch_evidence_sha256": batch_evidence_sha,
            "actor_provenance_sha256": actor_provenance_sha,
            "batch_id": batch_evidence["batch_id"],
            "media_join": media_join,
            "logprobs": logprob_evidence,
            "train_result": _jsonable(train_result),
            "finite_loss": float(loss_value.item()),
            "finite_positive_grad_norm": float(grad_norm_value.item()),
            "worker_baselines": baseline,
            "worker_optimizer_evidence": optimizer_evidence,
            "optimizer_update_count": 1,
            "restart_safe_replay": False,
        }
        update_result_sha = write_json_exclusive(
            run_root / "evidence" / "UPDATE_RESULT.json", update_result
        )

        start_stage("checkpoint")
        checkpoint_root = run_root / "checkpoint" / "step-000000"
        checkpoint_root.mkdir(mode=0o750)
        policy.save_checkpoint(
            str(checkpoint_root), optimizer_path=str(checkpoint_root)
        )
        policy.finalize_async_save()
        checkpoint_manifest = checkpoint_tree_manifest(checkpoint_root)
        checkpoint_manifest.update(
            {
                "run_id": args.run_id,
                "step_id": step_id,
                "update_result_sha256": update_result_sha,
            }
        )
        checkpoint_manifest_sha = write_json_exclusive(
            run_root / "evidence" / "CHECKPOINT_TREE.json",
            checkpoint_manifest,
        )
        checkpoint_tree_sha = checkpoint_manifest["checkpoint_tree_sha256"]
        checkpoint_joins = _scan_actor_evidence(
            policy.run_all_workers_single_data(
                "qualification_record_checkpoint_join",
                run_id=args.run_id,
                checkpoint_tree_sha256=checkpoint_tree_sha,
                controller_result_sha256=update_result_sha,
            ),
            world_size=projection["world_size"],
            run_id=args.run_id,
        )
        checkpoint_completed = True
        journal_summary = verify_optimizer_journal(
            run_root / "journal",
            run_id=args.run_id,
            world_size=projection["world_size"],
            checkpoint_join_required=True,
            expected_checkpoint_tree_sha256=checkpoint_tree_sha,
            expected_controller_result_sha256=update_result_sha,
        )
        journal_summary_sha = write_json_exclusive(
            run_root / "evidence" / "OPTIMIZER_JOURNAL_SUMMARY.json",
            journal_summary,
        )
        complete_stage(
            "checkpoint",
            {
                "checkpoint_tree_sha256": checkpoint_tree_sha,
                "journal_summary_sha256": journal_summary_sha,
            },
        )

        # A success marker is authoritative only after the step bulk and all
        # qualification workers have been cleanly released.
        start_stage("cleanup")
        policy.finish_step(meta)
        meta = None
        if policy.shutdown() is not True:
            raise QualificationError("policy worker shutdown did not succeed")
        policy = None

        if checkpoint_tree_manifest(checkpoint_root)[
            "checkpoint_tree_sha256"
        ] != checkpoint_tree_sha:
            raise QualificationError("checkpoint tree changed after worker shutdown")
        complete_stage(
            "cleanup",
            {"data_plane_cleanup": True, "worker_shutdown": True},
        )

        start_stage("result")
        final_result = {
            "format": RESULT_FORMAT,
            "status": "production-cp1-source-executed",
            "run_id": args.run_id,
            "source_stack_id": args.source_stack_id,
            "intent_sha256": intent_sha,
            "config_projection_sha256": projection_sha,
            "fixed_batch_evidence_sha256": batch_evidence_sha,
            "update_result_sha256": update_result_sha,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha,
            "checkpoint_tree_sha256": checkpoint_tree_sha,
            "optimizer_journal_summary_sha256": journal_summary_sha,
            "checkpoint_joins": checkpoint_joins,
            "driver_provenance": driver_provenance,
            "media_prev_ref_train_join": True,
            "loss_evidence": True,
            "gradient_evidence": True,
            "parameter_delta_evidence": True,
            "checkpoint_join": True,
            "data_plane_cleanup": True,
            "worker_shutdown": True,
            "checkpoint_reload_smoke": False,
            "context_parallel_size": 1,
            "data_plane_backend": projection["data_plane_backend"],
            "optimizer_update_count": 1,
            "restart_safe_replay": False,
            "remaining_release_holds": [
                "fresh-policy checkpoint reload was not executed",
                "optimizer outcome remains ambiguous across worker death before applied journal",
                "CP2 production parity is a separate lane",
            ],
        }
        final_sha = write_json_exclusive(run_root / "RESULT.json", final_result)
        write_json_exclusive(
            run_root / "RESULT.sha256.json",
            {"path": "RESULT.json", "sha256": final_sha},
        )
        complete_stage("result", {"result_sha256": final_sha})
        print(
            "NEMO_RL_PRODUCTION_NANO_TQ_FIXED_BATCH|"
            + json.dumps(
                {
                    "run_id": args.run_id,
                    "status": final_result["status"],
                    "result_sha256": final_sha,
                    "run_root": str(run_root),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        try:
            record_stage_event(
                run_root,
                run_id=args.run_id,
                stage=current_stage,
                status="failed",
                details={
                    "error_type": type(error).__name__,
                    "optimizer_dispatched": optimizer_dispatched,
                    "checkpoint_completed": checkpoint_completed,
                },
            )
        except Exception:
            traceback.print_exc()
        failure = {
            "format": "nemo-rl-production-nano-tq-failure-v1",
            "run_id": args.run_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "optimizer_dispatched": optimizer_dispatched,
            "checkpoint_completed": checkpoint_completed,
            "failed_stage": current_stage,
            "automatic_replay_allowed": False,
            "traceback": traceback.format_exc(),
        }
        try:
            write_json_exclusive(run_root / "FAILED.json", failure)
        except FileExistsError:
            pass
        raise
    finally:
        if policy is not None and meta is not None:
            try:
                policy.finish_step(meta)
            except Exception:
                traceback.print_exc()
        if policy is not None:
            try:
                policy.shutdown()
            except Exception:
                traceback.print_exc()


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run one production Nano-Omni CP1 update through TransferQueue"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--expected-source-root", required=True)
    parser.add_argument("--source-stack-id", required=True)
    parser.add_argument("--source-bundle-manifest", required=True)
    parser.add_argument("--source-bundle-manifest-sha256", required=True)
    parser.add_argument("--expected-model-root", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--expected-image-digest", required=True)
    parser.add_argument("--expected-image-fingerprint-sha256", required=True)
    parser.add_argument("--expected-driver-venv", required=True)
    parser.add_argument("--expected-actor-venv", required=True)
    parser.add_argument("--expected-python-version", required=True)
    parser.add_argument("--expected-ray-version", required=True)
    parser.add_argument("--expected-torch-version", required=True)
    parser.add_argument("--expected-cuda-compute-capability", required=True)
    parser.add_argument("--expected-num-nodes", type=int, required=True)
    parser.add_argument("--expected-gpus-per-node", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-tensor-parallel-size", type=int, required=True)
    parser.add_argument("--expected-pipeline-parallel-size", type=int, required=True)
    parser.add_argument(
        "--expected-context-parallel-size", type=int, choices=[1], required=True
    )
    parser.add_argument("--expected-expert-parallel-size", type=int, required=True)
    parser.add_argument("--expected-train-global-batch-size", type=int, required=True)
    parser.add_argument("--expected-train-micro-batch-size", type=int, required=True)
    parser.add_argument(
        "--expected-data-plane-backend",
        choices=["simple", "mooncake_cpu"],
        required=True,
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, overrides = parse_args(argv)
    return _run_production(args, overrides)


if __name__ == "__main__":
    raise SystemExit(main())
