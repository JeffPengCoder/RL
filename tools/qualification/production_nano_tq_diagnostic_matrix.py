# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Dependency-free plan and collector for Nano/TQ diagnostic lanes.

This module does not submit Slurm work.  A scheduler-owned launcher consumes
the immutable plan and starts every lane as an independent step/allocation, so
one nonzero lane never cancels an unrelated lane.  The trusted host collector
hashes the shared image, source tree, and model tree once before the matrix and
once after it.  Individual lanes consume that attestation plus immutable
stat/inode identities; they do not repeatedly hash the multi-gigabyte inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_FORMAT = "nemo-rl-production-nano-tq-diagnostic-matrix-plan-v1"
ATTESTATION_FORMAT = "nemo-rl-production-nano-tq-shared-attestation-v1"
LANE_RESULT_FORMAT = "nemo-rl-production-nano-tq-diagnostic-lane-result-v1"
MATRIX_RESULT_FORMAT = "nemo-rl-production-nano-tq-diagnostic-matrix-result-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")

SAFE_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "PYTHONFAULTHANDLER",
    "RAY_ADDRESS",
    "RAY_NAMESPACE",
    "SLURM_JOB_ID",
    "SLURM_STEP_ID",
    "TORCH_DISTRIBUTED_DEBUG",
}

LANE_DEFINITIONS = (
    {
        "lane_id": "actor-interpreter-smoke",
        "kind": "actor-api-smoke",
        "entrypoint_relative": (
            "tools/qualification/megatron_policy_worker_actor_smoke.py"
        ),
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "timeout_seconds": 900,
        "backend": None,
        "stage_enum": [
            "spawned",
            "imports",
            "module-realpaths",
            "cuda-probe",
            "complete",
        ],
    },
    {
        "lane_id": "tq-simple-roundtrip",
        "kind": "tq-api-smoke",
        "entrypoint_relative": "tests/unit/data_plane/test_packed_tensor_wire_tq.py",
        "pytest_selector": "simple",
        "nodes": 1,
        "gpus_per_node": 0,
        "cpus_per_task": 16,
        "timeout_seconds": 1200,
        "backend": "simple",
        "stage_enum": [
            "spawned",
            "ray-init",
            "tq-init",
            "put",
            "get",
            "verify",
            "teardown",
        ],
    },
    {
        "lane_id": "tq-mooncake-roundtrip",
        "kind": "tq-api-smoke",
        "entrypoint_relative": "tests/unit/data_plane/test_packed_tensor_wire_tq.py",
        "pytest_selector": "mooncake_cpu",
        "nodes": 1,
        "gpus_per_node": 0,
        "cpus_per_task": 16,
        "timeout_seconds": 1800,
        "backend": "mooncake_cpu",
        "fixed_service_ports": [50050, 50051],
        "stage_enum": [
            "spawned",
            "ray-init",
            "tq-init",
            "put",
            "get",
            "verify",
            "teardown",
        ],
    },
    {
        "lane_id": "nano-cp1-fixed-batch",
        "kind": "production-fixed-batch",
        "entrypoint_relative": (
            "tools/qualification/production_nano_tq_fixed_batch.py"
        ),
        "nodes": 1,
        "gpus_per_node": 8,
        "cpus_per_task": 128,
        "timeout_seconds": 10800,
        "backend": "mooncake_cpu",
        "stage_enum": [
            "config",
            "driver-provenance",
            "fixed-batch",
            "ray-policy-init",
            "actor-provenance",
            "tq-first-write",
            "prev-lp",
            "ref-lp",
            "optimizer-train",
            "media-join",
            "checkpoint",
            "cleanup",
            "result",
        ],
    },
)


class MatrixError(RuntimeError):
    """A fail-closed matrix plan or evidence violation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MatrixError(f"{field} must be a positive integer")
    return value


def _validate_artifact(
    value: Mapping[str, Any], *, field: str, root: bool
) -> dict[str, Any]:
    expected = {
        "path",
        "sha256",
        "bytes",
        "device",
        "inode",
        "mtime_ns",
        "verification_record_sha256",
    }
    if root:
        expected.add("tree_sha256")
        expected.add("manifest_sha256")
    if set(value) != expected:
        raise MatrixError(f"{field} attestation keys changed")
    result = dict(value)
    if not isinstance(result["path"], str) or not result["path"].startswith("/"):
        raise MatrixError(f"{field}.path must be absolute")
    for key in ("sha256", "verification_record_sha256"):
        if not isinstance(result[key], str) or SHA256_RE.fullmatch(result[key]) is None:
            raise MatrixError(f"{field}.{key} must be SHA256")
    if root:
        for key in ("tree_sha256", "manifest_sha256"):
            if not isinstance(result[key], str) or SHA256_RE.fullmatch(result[key]) is None:
                raise MatrixError(f"{field}.{key} must be SHA256")
    for key in ("bytes", "device", "inode", "mtime_ns"):
        _positive_int(result[key], f"{field}.{key}")
    return result


def validate_shared_attestation(
    value: Mapping[str, Any], *, phase: str | None = None
) -> dict[str, Any]:
    expected = {
        "format",
        "phase",
        "matrix_id",
        "time_ns",
        "image_digest",
        "image",
        "source",
        "model",
    }
    if set(value) != expected or value.get("format") != ATTESTATION_FORMAT:
        raise MatrixError("shared attestation schema changed")
    if value.get("phase") not in {"before", "after"}:
        raise MatrixError("shared attestation phase is invalid")
    if phase is not None and value["phase"] != phase:
        raise MatrixError(f"expected {phase} shared attestation")
    if ID_RE.fullmatch(str(value.get("matrix_id", ""))) is None:
        raise MatrixError("invalid matrix_id in shared attestation")
    _positive_int(value.get("time_ns"), "time_ns")
    if IMAGE_DIGEST_RE.fullmatch(str(value.get("image_digest", ""))) is None:
        raise MatrixError("image_digest must be a full immutable digest")
    result = dict(value)
    result["image"] = _validate_artifact(value["image"], field="image", root=False)
    result["source"] = _validate_artifact(value["source"], field="source", root=True)
    result["model"] = _validate_artifact(value["model"], field="model", root=True)
    return result


def _shared_identity(attestation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: attestation[key]
        for key in ("matrix_id", "image_digest", "image", "source", "model")
    }


def build_matrix_plan(
    *,
    before_attestation: Mapping[str, Any],
    output_parent: str,
    port_base: int,
) -> dict[str, Any]:
    before = validate_shared_attestation(before_attestation, phase="before")
    if not output_parent.startswith("/"):
        raise MatrixError("output_parent must be absolute")
    if port_base < 1024 or port_base + len(LANE_DEFINITIONS) * 100 > 49000:
        raise MatrixError("port_base is outside the sealed diagnostic range")
    matrix_id = before["matrix_id"]
    lanes = []
    for index, definition in enumerate(LANE_DEFINITIONS):
        lane = dict(definition)
        lane_id = lane["lane_id"]
        lane_root = f"{output_parent.rstrip('/')}/{matrix_id}/{lane_id}"
        lane.update(
            {
                "run_id": f"{matrix_id}-{lane_id}",
                "source_root": before["source"]["path"],
                "model_root": before["model"]["path"],
                "output_root": lane_root,
                "cache_root": f"{lane_root}/cache",
                "debug_root": f"{lane_root}/debug",
                "rank_log_root": f"{lane_root}/rank-logs",
                "ray_tmpdir": f"{lane_root}/ray-tmp",
                "ray_namespace": f"nrl-{matrix_id}-{lane_id}",
                "ray_port_low": port_base + index * 100,
                "ray_port_high": port_base + index * 100 + 79,
                "gpu_map": [str(gpu) for gpu in range(lane["gpus_per_node"])],
                "heartbeat_interval_seconds": 60,
                "stale_after_seconds": 1200,
                "failure_policy": "independent-no-cancel",
                "required_process_env": {
                    "PYTHONFAULTHANDLER": "1",
                    "RAY_NAMESPACE": f"nrl-{matrix_id}-{lane_id}",
                    "NCCL_DEBUG": "INFO" if lane["gpus_per_node"] else None,
                    "TORCH_DISTRIBUTED_DEBUG": (
                        "DETAIL" if lane["gpus_per_node"] else None
                    ),
                },
                "required_debug_fields": [
                    "stage",
                    "status",
                    "hostname",
                    "pid",
                    "slurm_job_id",
                    "slurm_step_id",
                    "image_digest",
                    "source_manifest_sha256",
                    "model_manifest_sha256",
                    "module_realpaths",
                    "dependency_versions",
                    "mountinfo",
                    "ray_namespace",
                    "ray_address",
                    "gpu_map",
                    "secret_env_presence",
                ],
            }
        )
        lanes.append(lane)
    plan = {
        "format": PLAN_FORMAT,
        "matrix_id": matrix_id,
        "before_attestation_sha256": sha256_json(before),
        "shared_identity_sha256": sha256_json(_shared_identity(before)),
        "shared_identity": _shared_identity(before),
        "lanes": lanes,
        "lane_failure_cancels_others": False,
        "shared_full_hash_policy": "once-before-and-once-after",
        "large_objects_must_not_be_rehashed_per_lane": True,
        "secrets": "presence-only",
        "shell_xtrace_allowed": False,
        "parallel_resource_floor": {
            "nodes": 4,
            "gpus": 9,
            "cpus": sum(lane["cpus_per_task"] for lane in lanes),
        },
        "resource_constraints": [
            "every lane uses an independent Slurm step or allocation",
            "every lane uses its own Ray namespace, port range, cache, output, and logs",
            "tq-mooncake-roundtrip needs an exclusive hostname for ports 50050/50051",
            "nano-cp1-fixed-batch needs one exclusive 8-H100 node",
            "lane timeout terminates only that lane; collection waits for all lanes",
        ],
    }
    return plan


def validate_lane_result(
    value: Mapping[str, Any], *, plan: Mapping[str, Any], lane: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "format",
        "matrix_id",
        "lane_id",
        "status",
        "terminal_stage",
        "started_ns",
        "finished_ns",
        "slurm_job_id",
        "slurm_step_id",
        "ray_namespace",
        "output_root",
        "cache_root",
        "gpu_map",
        "image_digest",
        "shared_identity_sha256",
        "safe_env",
        "secret_env_presence",
        "evidence_sha256",
    }
    if set(value) != required or value.get("format") != LANE_RESULT_FORMAT:
        raise MatrixError(f"lane result schema changed for {lane['lane_id']}")
    result = dict(value)
    exact = {
        "matrix_id": plan["matrix_id"],
        "lane_id": lane["lane_id"],
        "ray_namespace": lane["ray_namespace"],
        "output_root": lane["output_root"],
        "cache_root": lane["cache_root"],
        "gpu_map": lane["gpu_map"],
        "image_digest": plan["shared_identity"]["image_digest"],
        "shared_identity_sha256": plan["shared_identity_sha256"],
    }
    for key, expected in exact.items():
        if result.get(key) != expected:
            raise MatrixError(f"lane {lane['lane_id']} changed {key}")
    if result.get("status") not in {"passed", "failed", "timed-out"}:
        raise MatrixError(f"lane {lane['lane_id']} has invalid status")
    if result.get("terminal_stage") not in lane["stage_enum"]:
        raise MatrixError(f"lane {lane['lane_id']} has invalid terminal stage")
    started = _positive_int(result.get("started_ns"), "started_ns")
    finished = _positive_int(result.get("finished_ns"), "finished_ns")
    if finished < started:
        raise MatrixError(f"lane {lane['lane_id']} has reversed timestamps")
    if set(result.get("safe_env", {})) - SAFE_ENV_KEYS:
        raise MatrixError(f"lane {lane['lane_id']} recorded a non-allowlisted env")
    if not all(
        isinstance(present, bool)
        for present in result.get("secret_env_presence", {}).values()
    ):
        raise MatrixError("secret evidence must contain presence booleans only")
    if SHA256_RE.fullmatch(str(result.get("evidence_sha256", ""))) is None:
        raise MatrixError(f"lane {lane['lane_id']} has invalid evidence digest")
    return result


def collect_matrix_result(
    *,
    plan: Mapping[str, Any],
    before_attestation: Mapping[str, Any],
    after_attestation: Mapping[str, Any],
    lane_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before = validate_shared_attestation(before_attestation, phase="before")
    after = validate_shared_attestation(after_attestation, phase="after")
    if _shared_identity(before) != _shared_identity(after):
        raise MatrixError("shared image/source/model identity changed during matrix")
    if after["time_ns"] <= before["time_ns"]:
        raise MatrixError("after attestation does not follow before attestation")
    lanes = {lane["lane_id"]: lane for lane in plan["lanes"]}
    provided = {result.get("lane_id"): result for result in lane_results}
    if set(provided) != set(lanes):
        raise MatrixError("lane result set is incomplete or duplicated")
    validated = [
        validate_lane_result(provided[lane_id], plan=plan, lane=lanes[lane_id])
        for lane_id in sorted(lanes)
    ]
    statuses = {result["lane_id"]: result["status"] for result in validated}
    return {
        "format": MATRIX_RESULT_FORMAT,
        "matrix_id": plan["matrix_id"],
        "status": "passed" if all(value == "passed" for value in statuses.values()) else "failed",
        "before_attestation_sha256": sha256_json(before),
        "after_attestation_sha256": sha256_json(after),
        "shared_identity_sha256": plan["shared_identity_sha256"],
        "lane_statuses": statuses,
        "lane_failure_cancelled_others": False,
        "shared_identity_unchanged": True,
    }


def _read_json(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatrixError(f"expected JSON object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or collect Nano/TQ diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--before-attestation", required=True)
    plan.add_argument("--output-parent", required=True)
    plan.add_argument("--port-base", type=int, default=22000)
    plan.add_argument("--output", required=True)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--plan", required=True)
    collect.add_argument("--before-attestation", required=True)
    collect.add_argument("--after-attestation", required=True)
    collect.add_argument("--lane-result", action="append", required=True)
    collect.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        value = build_matrix_plan(
            before_attestation=_read_json(args.before_attestation),
            output_parent=args.output_parent,
            port_base=args.port_base,
        )
    else:
        value = collect_matrix_result(
            plan=_read_json(args.plan),
            before_attestation=_read_json(args.before_attestation),
            after_attestation=_read_json(args.after_attestation),
            lane_results=[_read_json(path) for path in args.lane_result],
        )
    digest = write_json_exclusive(Path(args.output), value)
    print(
        "NEMO_RL_PRODUCTION_NANO_TQ_MATRIX|"
        + json.dumps(
            {"command": args.command, "output": args.output, "sha256": digest},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
