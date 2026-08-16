#!/usr/bin/env python3
"""Dependency-free tests for the HF dynamic-code CPU lane runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("run_hf_dynamic_code_cpu_lanes.py")
SPEC = importlib.util.spec_from_file_location("hf_dynamic_code_cpu_lanes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class HFDynamicCodeCPULanesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)

    def tearDown(self) -> None:
        for directory, names, files in os.walk(self.base):
            directory_path = Path(directory)
            directory_path.chmod(0o755)
            for name in files:
                path = directory_path / name
                if not path.is_symlink():
                    path.chmod(0o644)
            for name in names:
                path = directory_path / name
                if not path.is_symlink():
                    path.chmod(0o755)
        self.temporary.cleanup()

    def _partial_tree(self, name: str = ".hf-modules.partial") -> Path:
        root = self.base / name
        (root / "transformers_modules" / "repo").mkdir(parents=True)
        (root / "__init__.py").write_bytes(b"")
        (root / "transformers_modules" / "__init__.py").write_bytes(b"")
        (root / "transformers_modules" / "repo" / "code.py").write_bytes(
            b"VALUE = 1\n"
        )
        return root

    def test_environment_evidence_never_contains_secret_values_or_hashes(self) -> None:
        evidence = RUNNER._environment_authority(
            {
                "HF_HUB_CACHE": "/sealed/hub",
                "HF_HUB_OFFLINE": "1",
                "HF_TOKEN": "top-secret-value",
                "API_KEY": "another-secret",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        encoded = json.dumps(evidence)
        self.assertNotIn("top-secret-value", encoded)
        self.assertNotIn("another-secret", encoded)
        self.assertNotIn("value_sha256", encoded)
        self.assertEqual(evidence["secret_presence"], ["API_KEY", "HF_TOKEN"])
        self.assertEqual(evidence["authorities"]["hf_hub_cache"], "/sealed/hub")

    def test_child_environment_does_not_propagate_hf_tokens(self) -> None:
        lane = self.base / "lane"
        lane.mkdir()
        hub = self.base / "hub"
        hub.mkdir()
        modules = lane / ".hf-modules.partial"
        with mock.patch.dict(
            os.environ,
            {"HF_TOKEN": "secret", "API_KEY": "secret", "PATH": "/usr/bin"},
            clear=True,
        ):
            environment = RUNNER._lane_environment(lane, hub, modules)
        self.assertNotIn("HF_TOKEN", environment)
        self.assertNotIn("API_KEY", environment)
        self.assertEqual(environment["HF_MODULES_CACHE"], str(modules))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")

    def test_mountinfo_parser_and_longest_mount_selection(self) -> None:
        payload = (
            b"1 0 8:1 / / rw - ext4 /dev/root rw\n"
            b"2 1 8:2 / /workspace/source ro,nosuid - lustre server:/x ro\n"
        )
        records = RUNNER._parse_mountinfo(payload)
        selected = RUNNER._mount_for_path(Path(__file__).resolve(), [
            {**records[0], "mount_point": "/"},
        ])
        self.assertEqual(selected["filesystem"], "ext4")
        self.assertIn("ro", records[1]["mount_options"])

    @unittest.skipUnless(
        sys.platform.startswith("linux"),
        "production publication primitive is qualified on Linux/Lustre",
    )
    def test_freeze_and_atomic_publication_preserve_inode_and_bytes(self) -> None:
        partial = self._partial_tree()
        frozen = RUNNER._freeze_tree(partial)
        RUNNER._require_frozen_manifest(frozen)
        destination = self.base / "hf-modules"
        publication = RUNNER._publish_frozen_tree(partial, destination)
        self.assertTrue(publication["same_root_inode"])
        self.assertFalse(partial.exists())
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o555)
        code = destination / "transformers_modules" / "repo" / "code.py"
        self.assertEqual(stat.S_IMODE(code.stat().st_mode), 0o444)
        self.assertEqual(code.read_bytes(), b"VALUE = 1\n")
        self.assertRegex(publication["manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_publication_refuses_existing_destination(self) -> None:
        partial = self._partial_tree()
        RUNNER._freeze_tree(partial)
        destination = self.base / "hf-modules"
        destination.mkdir()
        with self.assertRaisesRegex(RuntimeError, "must not already exist"):
            RUNNER._publish_frozen_tree(partial, destination)
        self.assertTrue(partial.is_dir())

    def test_negative_copy_is_private_and_mutable(self) -> None:
        source = self._partial_tree()
        RUNNER._freeze_tree(source)
        destination = self.base / ".negative.partial"
        RUNNER._copy_tree_mutable(source, destination)
        target = destination / "transformers_modules" / "repo" / "code.py"
        target.write_bytes(b"VALUE = 2\n")
        self.assertEqual(
            (source / "transformers_modules" / "repo" / "code.py").read_bytes(),
            b"VALUE = 1\n",
        )
        self.assertEqual(target.read_bytes(), b"VALUE = 2\n")

    def test_freeze_rejects_symlink(self) -> None:
        partial = self._partial_tree()
        link = partial / "escape.py"
        link.symlink_to(self.base / "outside.py")
        with self.assertRaisesRegex(RuntimeError, "non-regular leaf"):
            RUNNER._freeze_tree(partial)

    def test_diagnostic_matrix_has_disjoint_cpu_only_lanes(self) -> None:
        shared = self.base / "shared"
        shared.mkdir()
        files = []
        for name in ("evidence", "verifier", "runner", "preflight"):
            path = shared / name
            path.write_bytes(name.encode())
            files.append(path)
        hub = shared / "hub"
        model = shared / "model"
        hub.mkdir()
        model.mkdir()
        matrix = RUNNER._diagnostic_matrix(
            self.base / "output",
            hub,
            model,
            files[0],
            files[1],
            files[2],
            files[3],
        )
        self.assertEqual(matrix["cas_full_hash_count"], 1)
        self.assertEqual(matrix["hazards"], [])
        self.assertTrue(all(row["gpu_slots"] == 0 for row in matrix["lanes"]))
        self.assertEqual(len({row["writable_root"] for row in matrix["lanes"]}), 4)

    def test_diagnostic_matrix_rejects_output_under_shared_authority(self) -> None:
        shared = self.base / "shared"
        hub = shared
        model = self.base / "model"
        hub.mkdir()
        model.mkdir()
        evidence = self.base / "evidence"
        evidence.write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "overlaps shared authority"):
            RUNNER._diagnostic_matrix(
                shared / "output",
                hub,
                model,
                evidence,
                evidence,
                evidence,
                evidence,
            )

    def test_child_output_is_digest_only(self) -> None:
        result = RUNNER._run_child(
            [sys.executable, "-B", "-c", "print('diagnostic payload')"],
            dict(os.environ),
            self.base,
            10,
        )
        self.assertEqual(result["returncode"], 0)
        self.assertGreater(result["stdout_size"], 0)
        self.assertNotIn("stdout", result)
        self.assertNotIn("stderr", result)
        self.assertRegex(result["stdout_sha256"], r"^[0-9a-f]{64}$")

    def test_consumer_command_enforces_bytecode_and_safe_path(self) -> None:
        args = argparse.Namespace(
            lane="remote-positive",
            hub_cache_root=self.base / "hub",
            processor_model_path=self.base / "model",
            snapshot_verification=self.base / "source.json",
            cas_preflight=self.base / "preflight.json",
            modules_verifier=self.base / "verifier.py",
            repository="nvidia/C-RADIOv4-H",
            revision="b" * 40,
        )
        command = RUNNER._consumer_command(
            SCRIPT,
            args,
            self.base / "published",
            "pass",
            self.base / "result.json",
        )
        self.assertEqual(command[1:3], ["-B", "-P"])
        self.assertIn("--modules-cache-root", command)
        self.assertIn("--cas-preflight", command)

    def test_consumer_must_join_published_inode_and_tree(self) -> None:
        publication = {
            "content_manifest_sha256": "c" * 64,
            "destination_root_identity": [1, 2, 0, 0, 0, 0],
        }
        manifest = {
            "content_manifest_sha256": "c" * 64,
            "entries": [{"kind": "file", "path": "x", "sha256": "d" * 64}],
            "root_identity": [1, 2, 0, 0, 0, 0],
        }
        RUNNER._require_consumer_publication_join(
            publication, {"cache_before": manifest, "cache_after": dict(manifest)}
        )
        changed = dict(manifest)
        changed["root_identity"] = [1, 3, 0, 0, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "published root inode"):
            RUNNER._require_consumer_publication_join(
                publication, {"cache_before": changed, "cache_after": manifest}
            )

    def test_canonical_result_is_read_only_and_no_replace(self) -> None:
        body = {"format": "fixture", "status": "pass"}
        body["manifest_sha256"] = RUNNER._sha256(RUNNER._canonical_json(body))
        output = self.base / "result.json"
        RUNNER._write_json_noreplace(output, body)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        self.assertEqual(RUNNER._load_json(output, "fixture"), body)
        with self.assertRaises(FileExistsError):
            RUNNER._write_json_noreplace(output, body)


if __name__ == "__main__":
    unittest.main()
