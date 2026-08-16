#!/usr/bin/env python3
"""Dependency-free tests for the reusable Hugging Face cache verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("verify_hf_snapshot_mount.py")
SPEC = importlib.util.spec_from_file_location("stable_runtime_hf_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)

REVISION = "a" * 40
REMOTE_REVISION = "b" * 40
REMOTE_REPO = "nvidia/fixture-radio"


def _git_blob_name(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


class VerifyHFSnapshotMountTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.model_root = self.base / "models--nvidia--fixture"
        self.model_snapshot = self.model_root / "snapshots" / REVISION
        self.model_blobs = self.model_root / "blobs"
        self.model_snapshot.mkdir(parents=True)
        self.model_blobs.mkdir()
        self.remote_cache = self.base / "remote-cache"
        self.remote_root = self.remote_cache / "models--nvidia--fixture-radio"
        self.remote_snapshot = self.remote_root / "snapshots" / REMOTE_REVISION
        self.remote_blobs = self.remote_root / "blobs"
        self.remote_snapshot.mkdir(parents=True)
        self.remote_blobs.mkdir()
        (self.remote_root / "refs").mkdir()
        (self.remote_root / "refs" / "main").write_text(REMOTE_REVISION + "\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _add(
        self,
        snapshot: Path,
        blobs: Path,
        name: str,
        payload: bytes,
        *,
        sha256: bool = False,
    ) -> None:
        blob_name = hashlib.sha256(payload).hexdigest() if sha256 else _git_blob_name(payload)
        blob = blobs / blob_name
        if not blob.exists():
            blob.write_bytes(payload)
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        depth = len(destination.parent.relative_to(snapshot).parts)
        os.symlink("../" * (depth + 2) + f"blobs/{blob_name}", destination)

    def _populate(self) -> None:
        shards = ["weights/model-00001-of-00002.safetensors", "weights/model-00002-of-00002.safetensors"]
        weight_map = {"weight.0": shards[0], "weight.1": shards[1]}
        config = {
            "auto_map": {"AutoConfig": "configuration.FixtureConfig"},
            "vision_config": {
                "auto_map": {
                    "AutoConfig": f"{REMOTE_REPO}--hf_model.RadioConfig",
                    "AutoModel": f"{REMOTE_REPO}--hf_model.RadioModel",
                }
            },
        }
        processor = {
            "auto_map": {"AutoProcessor": "processing.FixtureProcessor"}
        }
        files = {
            "config.json": json.dumps(config).encode(),
            "preprocessor_config.json": json.dumps(processor).encode(),
            "model.safetensors.index.json": json.dumps({"weight_map": weight_map}).encode(),
            "configuration.py": b"from .configuration_helper import VALUE\nclass FixtureConfig: pass\n",
            "configuration_helper.py": b"VALUE = 1\n",
            "processing.py": b"class FixtureProcessor: pass\n",
        }
        for name, payload in files.items():
            self._add(self.model_snapshot, self.model_blobs, name, payload)
        for index, shard in enumerate(shards):
            self._add(
                self.model_snapshot,
                self.model_blobs,
                shard,
                f"weight-{index}".encode(),
                sha256=True,
            )
        remote_files = {
            "hf_model.py": b"from .radio_model import RadioBase\nclass RadioConfig: pass\nclass RadioModel: pass\n",
            "radio_model.py": b"from .utils import VALUE\nclass RadioBase: pass\n",
            "utils.py": b"VALUE = 1\n",
            "config.json": b"{}\n",
        }
        for name, payload in remote_files.items():
            self._add(self.remote_snapshot, self.remote_blobs, name, payload)

    def _verify(self, **overrides):
        remote_snapshot = VERIFIER._verify_snapshot(self.remote_root, REMOTE_REVISION)
        arguments = {
            "model_root": self.model_root,
            "revision": REVISION,
            "expected_shards": 2,
            "required_paths": (
                "config.json",
                "preprocessor_config.json",
                "model.safetensors.index.json",
                "processing.py",
            ),
            "auto_map_configs": ("config.json", "preprocessor_config.json"),
            "remote_code_cache_root": self.remote_cache,
            "remote_code_revisions": {REMOTE_REPO: REMOTE_REVISION},
            "expected_remote_code_manifests": {
                REMOTE_REPO: remote_snapshot["content_manifest_sha256"]
            },
        }
        arguments.update(overrides)
        return VERIFIER.verify(**arguments)

    def test_nested_local_and_remote_dynamic_code_passes(self) -> None:
        self._populate()
        result = self._verify()
        self.assertEqual(result["safetensors_shard_count"], 2)
        self.assertEqual(
            result["local_dynamic_module_closure"],
            ["configuration.py", "configuration_helper.py", "processing.py"],
        )
        self.assertEqual(result["remote_code"][0]["repository"], REMOTE_REPO)
        self.assertEqual(result["remote_code"][0]["main_ref_value"], REMOTE_REVISION)
        self.assertEqual(
            result["remote_code"][0]["main_ref_sha256"],
            hashlib.sha256((REMOTE_REVISION + "\n").encode()).hexdigest(),
        )
        self.assertEqual(
            result["remote_code"][0]["module_closure"],
            ["hf_model.py", "radio_model.py", "utils.py"],
        )
        self.assertRegex(result["snapshot_content_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_snapshot_only_mount_is_rejected(self) -> None:
        self._populate()
        with self.assertRaisesRegex(RuntimeError, "snapshot"):
            self._verify(model_root=self.model_snapshot)

    def test_model_root_symlink_is_rejected(self) -> None:
        self._populate()
        alias = self.base / "model-alias"
        alias.symlink_to(self.model_root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
            self._verify(model_root=alias)

    def test_in_root_non_blob_target_is_rejected(self) -> None:
        self._populate()
        payload = b"wrong namespace"
        digest = _git_blob_name(payload)
        other = self.model_root / "other"
        other.mkdir()
        (other / digest).write_bytes(payload)
        os.symlink(f"../../other/{digest}", self.model_snapshot / "escape.py")
        with self.assertRaisesRegex(RuntimeError, "exact blobs namespace"):
            self._verify()

    def test_blob_digest_mismatch_is_rejected(self) -> None:
        self._populate()
        target = next(self.model_blobs.iterdir())
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            self._verify()

    def test_unpinned_remote_reference_is_rejected(self) -> None:
        self._populate()
        with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
            self._verify(remote_code_revisions={})

    def test_dangling_remote_manifest_is_rejected_without_remote_auto_map(self) -> None:
        self._populate()
        config = {"auto_map": {"AutoConfig": "configuration.FixtureConfig"}}
        self._add(
            self.model_snapshot,
            self.model_blobs,
            "local_only_config.json",
            json.dumps(config).encode(),
        )
        with self.assertRaisesRegex(RuntimeError, "expected manifests"):
            self._verify(
                auto_map_configs=("local_only_config.json",),
                remote_code_revisions={},
                expected_remote_code_manifests={REMOTE_REPO: "0" * 64},
            )

    def test_missing_remote_transitive_module_is_rejected(self) -> None:
        self._populate()
        (self.remote_snapshot / "utils.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "relative dynamic-module import is absent"):
            self._verify()

    def test_missing_from_dot_import_is_rejected(self) -> None:
        self._populate()
        (self.remote_snapshot / "radio_model.py").unlink()
        self._add(
            self.remote_snapshot,
            self.remote_blobs,
            "radio_model.py",
            b"from . import missing\nclass RadioBase: pass\n",
        )
        with self.assertRaisesRegex(RuntimeError, "relative dynamic-module import is absent"):
            self._verify()

    def test_duplicate_json_key_is_rejected(self) -> None:
        self._populate()
        (self.model_snapshot / "config.json").unlink()
        duplicate = b'{"auto_map":{"AutoConfig":"configuration.FixtureConfig"},"auto_map":{}}'
        self._add(self.model_snapshot, self.model_blobs, "config.json", duplicate)
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON object key"):
            self._verify()

    def test_nonfinite_json_constant_is_rejected(self) -> None:
        self._populate()
        (self.model_snapshot / "config.json").unlink()
        nonfinite = b'{"value":NaN,"auto_map":{"AutoConfig":"configuration.FixtureConfig"}}'
        self._add(self.model_snapshot, self.model_blobs, "config.json", nonfinite)
        with self.assertRaisesRegex(RuntimeError, "non-finite JSON constant"):
            self._verify()

    def test_unindexed_safetensors_file_is_rejected(self) -> None:
        self._populate()
        self._add(
            self.model_snapshot,
            self.model_blobs,
            "weights/stale.safetensors",
            b"stale",
            sha256=True,
        )
        with self.assertRaisesRegex(RuntimeError, "do not exactly match the index"):
            self._verify()

    def test_remote_main_ref_extra_whitespace_is_rejected(self) -> None:
        self._populate()
        (self.remote_root / "refs" / "main").write_text(REMOTE_REVISION + "\n\n")
        with self.assertRaisesRegex(RuntimeError, "main ref is not pinned"):
            self._verify()

    def test_expected_remote_manifest_mismatch_is_rejected(self) -> None:
        self._populate()
        with self.assertRaisesRegex(RuntimeError, "remote-code snapshot manifest mismatch"):
            self._verify(expected_remote_code_manifests={REMOTE_REPO: "0" * 64})

    def test_offline_runtime_environment_is_cross_bound(self) -> None:
        self._populate()
        environment = {
            "HF_HUB_CACHE": str(self.remote_cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            result = self._verify(require_offline_env=True)
        self.assertEqual(
            result["offline_environment"]["hf_hub_cache"],
            str(self.remote_cache.resolve()),
        )

    def test_offline_runtime_environment_rejects_wrong_cache(self) -> None:
        self._populate()
        wrong = self.base / "wrong-cache"
        wrong.mkdir()
        environment = {
            "HF_HUB_CACHE": str(wrong),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "does not select"):
                self._verify(require_offline_env=True)

    def test_expected_snapshot_manifest_mismatch_is_rejected(self) -> None:
        self._populate()
        with self.assertRaisesRegex(RuntimeError, "snapshot manifest mismatch"):
            self._verify(expected_snapshot_manifest_sha256="0" * 64)

    def test_output_publication_is_no_replace(self) -> None:
        output = self.base / "result.json"
        output.symlink_to(self.base / "missing")
        with self.assertRaises(FileExistsError):
            VERIFIER._write_atomic_noreplace(output, b"{}\n")
        self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
