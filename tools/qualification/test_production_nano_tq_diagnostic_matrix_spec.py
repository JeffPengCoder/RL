# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Isolated stdlib tests for the Nano/TQ diagnostic matrix contract."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path


MATRIX_PATH = Path(__file__).with_name("production_nano_tq_diagnostic_matrix.py")
MATRIX_PATH = MATRIX_PATH.resolve(strict=True)
spec = importlib.util.spec_from_file_location("_nano_tq_matrix_under_test", MATRIX_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load matrix module: {MATRIX_PATH}")
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


class ProductionNanoTQMatrixTest(unittest.TestCase):
    @staticmethod
    def _artifact(prefix: str, *, root: bool) -> dict:
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        second = hashlib.sha256((prefix + "-second").encode("utf-8")).hexdigest()
        third = hashlib.sha256((prefix + "-third").encode("utf-8")).hexdigest()
        value = {
            "path": f"/lustre/{prefix}",
            "sha256": digest,
            "bytes": 1024,
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
            "verification_record_sha256": second,
        }
        if root:
            value.update(
                {
                    "tree_sha256": second,
                    "manifest_sha256": third,
                }
            )
        return value

    def _attestation(self, phase: str, *, time_ns: int) -> dict:
        return {
            "format": matrix.ATTESTATION_FORMAT,
            "phase": phase,
            "matrix_id": "nano-matrix-0001",
            "time_ns": time_ns,
            "image_digest": "sha256:" + "a" * 64,
            "image": self._artifact("abc-image", root=False),
            "source": self._artifact("def-source", root=True),
            "model": self._artifact("ghi-model", root=True),
        }

    @staticmethod
    def _lane_result(plan: dict, lane: dict) -> dict:
        return {
            "format": matrix.LANE_RESULT_FORMAT,
            "matrix_id": plan["matrix_id"],
            "lane_id": lane["lane_id"],
            "status": "passed",
            "terminal_stage": lane["stage_enum"][-1],
            "started_ns": 10,
            "finished_ns": 20,
            "slurm_job_id": "123",
            "slurm_step_id": "123.4",
            "ray_namespace": lane["ray_namespace"],
            "output_root": lane["output_root"],
            "cache_root": lane["cache_root"],
            "gpu_map": lane["gpu_map"],
            "image_digest": plan["shared_identity"]["image_digest"],
            "shared_identity_sha256": plan["shared_identity_sha256"],
            "safe_env": {
                "PYTHONFAULTHANDLER": "1",
                "RAY_NAMESPACE": lane["ray_namespace"],
            },
            "secret_env_presence": {"HF_TOKEN": False, "WANDB_API_KEY": True},
            "evidence_sha256": "e" * 64,
        }

    def test_isolated_exact_file_loader(self) -> None:
        self.assertEqual(Path(matrix.__file__).resolve(strict=True), MATRIX_PATH)
        self.assertRegex(hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest(), r"^[0-9a-f]{64}$")

    def test_plan_separates_every_lane_resource(self) -> None:
        before = self._attestation("before", time_ns=1)
        plan = matrix.build_matrix_plan(
            before_attestation=before,
            output_parent="/workspace/output/diagnostics",
            port_base=22000,
        )
        lanes = plan["lanes"]
        self.assertEqual(len(lanes), 4)
        for key in (
            "output_root",
            "cache_root",
            "debug_root",
            "rank_log_root",
            "ray_tmpdir",
            "ray_namespace",
            "ray_port_low",
        ):
            self.assertEqual(len({lane[key] for lane in lanes}), len(lanes))
        self.assertFalse(plan["lane_failure_cancels_others"])
        self.assertEqual(plan["shared_full_hash_policy"], "once-before-and-once-after")
        self.assertFalse(plan["shell_xtrace_allowed"])
        mooncake = next(lane for lane in lanes if lane["lane_id"] == "tq-mooncake-roundtrip")
        self.assertEqual(mooncake["fixed_service_ports"], [50050, 50051])

    def test_collect_waits_for_all_lanes_and_binds_post_attestation(self) -> None:
        before = self._attestation("before", time_ns=1)
        after = self._attestation("after", time_ns=2)
        plan = matrix.build_matrix_plan(
            before_attestation=before,
            output_parent="/workspace/output/diagnostics",
            port_base=22000,
        )
        lane_results = [self._lane_result(plan, lane) for lane in plan["lanes"]]
        lane_results[1]["status"] = "failed"
        result = matrix.collect_matrix_result(
            plan=plan,
            before_attestation=before,
            after_attestation=after,
            lane_results=lane_results,
        )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["lane_failure_cancelled_others"])
        self.assertTrue(result["shared_identity_unchanged"])

        with self.assertRaisesRegex(matrix.MatrixError, "incomplete"):
            matrix.collect_matrix_result(
                plan=plan,
                before_attestation=before,
                after_attestation=after,
                lane_results=lane_results[:-1],
            )

        changed = self._attestation("after", time_ns=2)
        changed["model"]["tree_sha256"] = "f" * 64
        with self.assertRaisesRegex(matrix.MatrixError, "identity changed"):
            matrix.collect_matrix_result(
                plan=plan,
                before_attestation=before,
                after_attestation=changed,
                lane_results=lane_results,
            )

    def test_lane_result_rejects_env_dump_and_secret_values(self) -> None:
        before = self._attestation("before", time_ns=1)
        plan = matrix.build_matrix_plan(
            before_attestation=before,
            output_parent="/workspace/output/diagnostics",
            port_base=22000,
        )
        lane = plan["lanes"][0]
        result = self._lane_result(plan, lane)
        result["safe_env"]["PATH"] = "/untrusted"
        with self.assertRaisesRegex(matrix.MatrixError, "non-allowlisted"):
            matrix.validate_lane_result(result, plan=plan, lane=lane)

        result = self._lane_result(plan, lane)
        result["secret_env_presence"]["HF_TOKEN"] = "secret-value"
        with self.assertRaisesRegex(matrix.MatrixError, "presence booleans"):
            matrix.validate_lane_result(result, plan=plan, lane=lane)


if __name__ == "__main__":
    unittest.main(verbosity=2)
