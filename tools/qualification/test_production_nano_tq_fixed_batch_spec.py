# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Dependency-free contract tests for the production Nano/TQ harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


HARNESS_PATH = Path(__file__).with_name("production_nano_tq_fixed_batch.py")
if HARNESS_PATH.is_symlink() or not HARNESS_PATH.is_file():
    raise RuntimeError(f"qualification harness is missing or symlinked: {HARNESS_PATH}")
HARNESS_PATH = HARNESS_PATH.resolve(strict=True)
HARNESS_SHA256 = hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()
expected_harness_sha256 = os.environ.get(
    "NEMO_RL_EXPECTED_PRODUCTION_NANO_TQ_DRIVER_SHA256"
)
if (
    expected_harness_sha256 is not None
    and expected_harness_sha256 != HARNESS_SHA256
):
    raise RuntimeError(
        "qualification harness SHA256 changed: "
        f"{HARNESS_SHA256} != {expected_harness_sha256}"
    )
spec = importlib.util.spec_from_file_location(
    "_production_nano_tq_fixed_batch_under_test", HARNESS_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load qualification harness: {HARNESS_PATH}")
harness = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = harness
spec.loader.exec_module(harness)

QualificationError = harness.QualificationError
checkpoint_tree_manifest = harness.checkpoint_tree_manifest
create_fresh_run_root = harness.create_fresh_run_root
sha256_json = harness.sha256_json
validate_config_projection = harness.validate_config_projection
validate_media_trace_join = harness.validate_media_trace_join
validate_static_contract = harness.validate_static_contract
validated_source_pythonpath = harness.validated_source_pythonpath
verify_optimizer_journal = harness.verify_optimizer_journal
write_json_exclusive = harness.write_json_exclusive


class ProductionNanoTQContractTest(unittest.TestCase):
    def _contract(self) -> dict:
        return {
            "run_id": "nano-tq-test-0001",
            "source_stack_id": "source-stack-test",
            "source_bundle_manifest_sha256": "1" * 64,
            "model_manifest_sha256": "2" * 64,
            "expected_image_digest": "sha256:" + "e" * 64,
            "expected_image_fingerprint_sha256": "f" * 64,
            "expected_driver_venv": "/opt/nemo_rl_venv",
            "expected_actor_venv": "/opt/ray_venvs/mcore",
            "expected_python_version": "3.13.13",
            "expected_ray_version": "2.55.1",
            "expected_torch_version": "2.11.0",
            "expected_cuda_compute_capability": "9.0",
            "expected_num_nodes": 1,
            "expected_gpus_per_node": 8,
            "expected_world_size": 8,
            "expected_tensor_parallel_size": 8,
            "expected_pipeline_parallel_size": 1,
            "expected_context_parallel_size": 1,
            "expected_expert_parallel_size": 8,
            "expected_train_global_batch_size": 8,
            "expected_train_micro_batch_size": 1,
            "expected_sequence_packing_enabled": False,
            "expected_data_plane_backend": "mooncake_cpu",
            "checkpoint_required": True,
            "restart_safe_replay": False,
        }

    def _projection(self) -> dict:
        return {
            "data_plane_enabled": True,
            "data_plane_impl": "transfer_queue",
            "data_plane_backend": "mooncake_cpu",
            "policy_backend": "megatron",
            "is_vlm": True,
            "router_replay_enabled": False,
            "tokenizer_use_fastokens": False,
            "generation_refit_transport": None,
            "num_nodes": 1,
            "gpus_per_node": 8,
            "context_parallel_size": 1,
            "world_size": 8,
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 8,
            "train_global_batch_size": 8,
            "data_parallel_size": 1,
            "reference_policy_kl_penalty": 0.01,
            "dynamic_batching_enabled": False,
            "mtp_num_layers": None,
            "fused_linear_logprobs": False,
            "virtual_pipeline_size": None,
            "train_micro_batch_size": 1,
            "sequence_packing_enabled": False,
        }

    def test_isolated_loader_uses_exact_sibling(self) -> None:
        self.assertEqual(Path(harness.__file__).resolve(strict=True), HARNESS_PATH)
        self.assertRegex(HARNESS_SHA256, r"^[0-9a-f]{64}$")

    def test_static_contract_is_exact_and_cp1_only(self) -> None:
        contract = self._contract()
        self.assertEqual(validate_static_contract(contract), contract)

        changed = dict(contract)
        changed["unexpected"] = True
        with self.assertRaisesRegex(QualificationError, "keys changed"):
            validate_static_contract(changed)

        cp2 = dict(contract)
        cp2["expected_context_parallel_size"] = 2
        with self.assertRaisesRegex(QualificationError, "CP1-only"):
            validate_static_contract(cp2)

        floating_ray = dict(contract)
        floating_ray["expected_ray_version"] = ">=2.55.1"
        with self.assertRaisesRegex(QualificationError, "exact X.Y.Z"):
            validate_static_contract(floating_ray)

    def test_config_projection_fails_closed(self) -> None:
        projection = self._projection()
        self.assertEqual(
            validate_config_projection(projection, self._contract()), projection
        )

        router_replay = dict(projection)
        router_replay["router_replay_enabled"] = True
        with self.assertRaisesRegex(QualificationError, "route authority"):
            validate_config_projection(router_replay, self._contract())

        dp2 = dict(projection)
        dp2["data_parallel_size"] = 2
        with self.assertRaisesRegex(QualificationError, "data_parallel_size=1"):
            validate_config_projection(dp2, self._contract())

        packed = dict(projection)
        packed["sequence_packing_enabled"] = True
        with self.assertRaisesRegex(QualificationError, "sequence_packing_enabled"):
            validate_config_projection(packed, self._contract())

        wrong_tp = dict(projection)
        wrong_tp["tensor_parallel_size"] = 4
        with self.assertRaisesRegex(QualificationError, "tensor_parallel_size"):
            validate_config_projection(wrong_tp, self._contract())

        fastokens = dict(projection)
        fastokens["tokenizer_use_fastokens"] = True
        with self.assertRaisesRegex(QualificationError, "Fastokens"):
            validate_config_projection(fastokens, self._contract())

        refit = dict(projection)
        refit["generation_refit_transport"] = "nixl"
        with self.assertRaisesRegex(QualificationError, "refit transports"):
            validate_config_projection(refit, self._contract())

    def test_fresh_run_root_refuses_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = create_fresh_run_root(parent, "nano-tq-run-0001")
            self.assertTrue((root / "journal").is_dir())
            with self.assertRaisesRegex(QualificationError, "replay is forbidden"):
                create_fresh_run_root(parent, "nano-tq-run-0001")

    def test_source_pythonpath_is_complete_and_source_owned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = (
                root
                / "3rdparty"
                / "Megatron-Bridge-workspace"
                / "Megatron-Bridge"
            )
            (bridge / "src").mkdir(parents=True)
            (bridge / "3rdparty" / "Megatron-LM").mkdir(parents=True)
            observed = validated_source_pythonpath(root).split(os.pathsep)
            self.assertEqual(
                observed,
                [
                    str(root.resolve()),
                    str((bridge / "src").resolve()),
                    str((bridge / "3rdparty" / "Megatron-LM").resolve()),
                ],
            )

            (bridge / "src").rmdir()
            with self.assertRaisesRegex(QualificationError, "source_pythonpath"):
                validated_source_pythonpath(root)

    @staticmethod
    def _media_fixture() -> tuple[dict, list[dict]]:
        sample_ids = ["sample-0", "sample-1"]
        raw_sha = {
            "sample-0": "3" * 64,
            "sample-1": "4" * 64,
        }
        row_sha = {
            sample_id: sha256_json(
                {
                    "dtype": "torch.float32",
                    "shape": [1, 3, 4, 4],
                    "bytes_sha256": raw_sha[sample_id],
                }
            )
            for sample_id in sample_ids
        }
        schema = {
            "wire_schema_id": "5" * 64,
            "entries": [
                {
                    "logical_key": "pixel_values",
                    "row_sha256_by_sample_id": row_sha,
                }
            ],
        }
        records = []
        for stage in ("prev_lp", "ref_lp", "train"):
            for sample_id in sample_ids:
                for rank in range(2):
                    records.append(
                        {
                            "event": "tq_fetch_sample",
                            "stage": stage,
                            "key": sample_id,
                            "rank": rank,
                            "media_wire_schema_id": schema["wire_schema_id"],
                            "packed_tensor_media": {
                                "pixel_values": {
                                    "dtype": "torch.float32",
                                    "shape": [1, 3, 4, 4],
                                    "sha256": raw_sha[sample_id],
                                }
                            },
                        }
                    )
        return schema, records

    def test_media_join_requires_all_stages_ranks_and_composite_digest(self) -> None:
        schema, records = self._media_fixture()
        result = validate_media_trace_join(
            schema=schema,
            records=records,
            sample_ids=["sample-0", "sample-1"],
            expected_world_size=2,
        )
        self.assertEqual(result["rank_count"], 2)
        self.assertEqual(result["stages"], ["prev_lp", "ref_lp", "train"])

        incomplete = records[:-1]
        with self.assertRaisesRegex(QualificationError, "incomplete"):
            validate_media_trace_join(
                schema=schema,
                records=incomplete,
                sample_ids=["sample-0", "sample-1"],
                expected_world_size=2,
            )

        corrupt = json.loads(json.dumps(records))
        corrupt[0]["packed_tensor_media"]["pixel_values"]["sha256"] = "6" * 64
        with self.assertRaisesRegex(QualificationError, "digest mismatch"):
            validate_media_trace_join(
                schema=schema,
                records=corrupt,
                sample_ids=["sample-0", "sample-1"],
                expected_world_size=2,
            )

    @staticmethod
    def _write_rank_journal(root: Path, rank: int) -> None:
        identity = {
            "format": "nemo-rl-production-nano-tq-optimizer-journal-v1",
            "run_id": "nano-tq-test-0001",
            "rank": rank,
            "step_id": "step-0",
            "partition_id": "partition-0",
            "sample_ids": ["sample-0", "sample-1"],
            "media_wire_schema_id": "7" * 64,
        }
        phases = {
            "baseline": {
                **identity,
                "phase": "baseline",
                "parameter_state": {"parameter_sha256": "8" * 64},
            },
            "optimizer-dispatched": {
                **identity,
                "phase": "optimizer-dispatched",
                "restart_safe_replay": False,
            },
            "optimizer-applied": {
                **identity,
                "phase": "optimizer-applied",
                "parameter_delta": True,
                "restart_safe_replay": False,
                "post_parameter_state": {
                    "parameter_sha256": "9" * 64,
                    "gradient_sha256": "a" * 64,
                    "gradients_finite": True,
                    "gradients_nonzero": True,
                },
            },
            "checkpoint-joined": {
                **identity,
                "phase": "checkpoint-joined",
                "checkpoint_tree_sha256": "b" * 64,
                "controller_result_sha256": "c" * 64,
            },
        }
        rank_root = root / f"rank-{rank:05d}"
        for phase, payload in phases.items():
            write_json_exclusive(rank_root / f"{phase}.json", payload)

    def test_optimizer_journal_requires_applied_delta_and_checkpoint_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(2):
                self._write_rank_journal(root, rank)
            result = verify_optimizer_journal(
                root,
                run_id="nano-tq-test-0001",
                world_size=2,
                checkpoint_join_required=True,
                expected_checkpoint_tree_sha256="b" * 64,
                expected_controller_result_sha256="c" * 64,
            )
            self.assertFalse(result["restart_safe_replay"])
            self.assertEqual(len(result["rank_records"]), 2)

            write_json_exclusive(
                root / "rank-00001" / "optimizer-outcome-ambiguous.json",
                {"format": "failure"},
            )
            with self.assertRaisesRegex(QualificationError, "terminal failure"):
                verify_optimizer_journal(
                    root,
                    run_id="nano-tq-test-0001",
                    world_size=2,
                    checkpoint_join_required=True,
                    expected_checkpoint_tree_sha256="b" * 64,
                    expected_controller_result_sha256="c" * 64,
                )

            (root / "rank-00001" / "optimizer-outcome-ambiguous.json").unlink()
            joined = root / "rank-00001" / "checkpoint-joined.json"
            payload = json.loads(joined.read_text(encoding="utf-8"))
            payload["checkpoint_tree_sha256"] = "d" * 64
            joined.chmod(0o644)
            joined.write_bytes(harness.canonical_json_bytes(payload) + b"\n")
            with self.assertRaisesRegex(QualificationError, "tree join changed"):
                verify_optimizer_journal(
                    root,
                    run_id="nano-tq-test-0001",
                    world_size=2,
                    checkpoint_join_required=True,
                    expected_checkpoint_tree_sha256="b" * 64,
                    expected_controller_result_sha256="c" * 64,
                )

    def test_checkpoint_manifest_hashes_regular_files_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"weights")
            manifest = checkpoint_tree_manifest(root)
            self.assertEqual(manifest["file_count"], 1)
            self.assertEqual(manifest["total_bytes"], 7)

            (root / "escape").symlink_to(root / "weights.bin")
            with self.assertRaisesRegex(QualificationError, "symlink"):
                checkpoint_tree_manifest(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
