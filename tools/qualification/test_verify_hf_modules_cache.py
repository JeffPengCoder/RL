#!/usr/bin/env python3
"""Dependency-free tests for the Transformers modules-cache verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock


SCRIPT = Path(__file__).with_name("verify_hf_modules_cache.py")
SPEC = importlib.util.spec_from_file_location("hf_modules_cache_verifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)

REPOSITORY = "nvidia/C-RADIOv4-H"
REVISION = "b" * 40


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _git_blob_name(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode())
    digest.update(payload)
    return digest.hexdigest()


class VerifyHFModulesCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.hub_cache = self.base / "hub"
        self.repo_root = self.hub_cache / "models--nvidia--C-RADIOv4-H"
        self.snapshot = self.repo_root / "snapshots" / REVISION
        self.blobs = self.repo_root / "blobs"
        self.snapshot.mkdir(parents=True)
        self.blobs.mkdir()
        self.modules = self.base / "modules"
        self.modules.mkdir()
        self.source_payloads = {
            "config.json": b"{}\n",
            "preprocessor_config.json": b"{}\n",
            "hf_model.py": b"from .radio_model import RadioModel\n",
            "radio_model.py": b"from .utils import VALUE\nclass RadioModel: pass\n",
            "utils.py": b"VALUE = 1\n",
        }
        for name, payload in self.source_payloads.items():
            self._add_hub_file(name, payload)
        self.closure = ["hf_model.py", "radio_model.py", "utils.py"]
        self.entries = ["hf_model.py"]
        self._build_generated_cache()
        self.source_evidence = self.base / "source-verification.json"
        self.observation = self.base / "execution-observation.json"
        self._write_source_evidence()
        self.environment = {
            "HF_HUB_CACHE": str(self.hub_cache.resolve()),
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": str(self.modules.resolve()),
            "TRANSFORMERS_OFFLINE": "1",
        }
        self._write_observation()
        self._freeze_modules()

    def tearDown(self) -> None:
        self._thaw_modules()
        self.temporary.cleanup()

    def _freeze_modules(self) -> None:
        for directory, names, files in os.walk(self.modules, topdown=False):
            directory_path = Path(directory)
            for name in files:
                path = directory_path / name
                if not path.is_symlink():
                    path.chmod(0o444)
            for name in names:
                path = directory_path / name
                if not path.is_symlink():
                    path.chmod(0o555)
            directory_path.chmod(0o555)

    def _thaw_modules(self) -> None:
        if not self.modules.exists():
            return
        for directory, names, files in os.walk(self.modules):
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

    def _add_hub_file(self, name: str, payload: bytes) -> None:
        digest = _git_blob_name(payload)
        blob = self.blobs / digest
        blob.write_bytes(payload)
        destination = self.snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        depth = len(destination.parent.relative_to(self.snapshot).parts)
        os.symlink("../" * (depth + 2) + f"blobs/{digest}", destination)

    @property
    def repo_relative(self) -> PurePosixPath:
        return VERIFIER._generated_repository_relative(REPOSITORY)

    @property
    def revision_relative(self) -> PurePosixPath:
        return self.repo_relative / REVISION

    @property
    def prefix(self) -> str:
        return ".".join(self.revision_relative.parts)

    def _build_generated_cache(self) -> None:
        (self.modules / "__init__.py").write_bytes(b"")
        current = self.modules
        for part in self.revision_relative.parts:
            current = current / part
            current.mkdir()
            (current / "__init__.py").write_bytes(b"")
        for name in self.closure:
            destination = self.modules / self.revision_relative / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.source_payloads[name])

    def _snapshot_manifest(self) -> str:
        records = []
        for name, payload in sorted(self.source_payloads.items()):
            digest = _git_blob_name(payload)
            records.append(
                {
                    "blob_algorithm": "git-blob-sha1",
                    "blob_digest": digest,
                    "link": f"../../blobs/{digest}",
                    "path": name,
                    "size": len(payload),
                    "target_relative": f"blobs/{digest}",
                }
            )
        content = {"files": records, "revision": REVISION}
        return hashlib.sha256(_canonical(content)).hexdigest()

    def _write_source_evidence(self) -> None:
        body = {
            "format": VERIFIER.SOURCE_FORMAT,
            "offline_environment": {
                "hf_hub_cache": str(self.hub_cache.resolve()),
                "hf_hub_offline": "1",
                "transformers_offline": "1",
            },
            "remote_code": [
                {
                    "content_manifest_sha256": self._snapshot_manifest(),
                    "entry_modules": self.entries,
                    "module_closure": self.closure,
                    "repository": REPOSITORY,
                    "revision": REVISION,
                }
            ],
        }
        body["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        self.source_evidence.write_bytes(_canonical(body))

    def _capture(self, *, wrong_file: Path | None = None):
        module_name = self.prefix + ".hf_model"
        module = types.ModuleType(module_name)
        module.__file__ = str(
            wrong_file.resolve()
            if wrong_file is not None
            else (self.modules / self.revision_relative / "hf_model.py").resolve()
        )
        path = [str(self.modules.resolve()), *sys.path]
        with mock.patch.dict(os.environ, self.environment, clear=False):
            with mock.patch.object(sys, "path", path):
                with mock.patch.dict(sys.modules, {module_name: module}, clear=False):
                    return VERIFIER.capture_execution_observation(
                        self.modules,
                        REPOSITORY,
                        REVISION,
                        self.closure,
                        self.entries,
                    )

    def _write_observation(self, *, wrong_file: Path | None = None) -> None:
        observation = self._capture(wrong_file=wrong_file)
        self.observation.write_bytes(_canonical(observation))

    def _verify(self, **overrides):
        arguments = {
            "modules_cache_root": self.modules,
            "hub_cache_root": self.hub_cache,
            "snapshot_verification_path": self.source_evidence,
            "execution_observation_path": self.observation,
            "repository": REPOSITORY,
            "revision": REVISION,
            "transformers_version": VERIFIER.TRANSFORMERS_VERSION,
        }
        arguments.update(overrides)
        with mock.patch.dict(os.environ, self.environment, clear=False):
            return VERIFIER.verify(**arguments)

    def test_exact_generated_cache_and_execution_pass(self) -> None:
        result = self._verify()
        self.assertEqual(result["remote_source_count"], 3)
        self.assertEqual(result["repository"], REPOSITORY)
        self.assertEqual(result["layout"], "transformers-5.8.1-remote-code")
        executed = [
            row for row in result["executed_modules"] if row["kind"] == "remote-source"
        ]
        self.assertEqual([row["source_relative"] for row in executed], ["hf_model.py"])
        self.assertRegex(result["manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_generated_source_byte_mismatch_is_rejected(self) -> None:
        self._thaw_modules()
        target = self.modules / self.revision_relative / "utils.py"
        target.write_bytes(b"VALUE = 2\n")
        self._freeze_modules()
        with self.assertRaisesRegex(RuntimeError, "do not match verified Hub CAS"):
            self._verify()

    def test_extra_generated_file_is_rejected(self) -> None:
        self._thaw_modules()
        (self.modules / self.revision_relative / "stale.py").write_text("STALE = 1\n")
        self._freeze_modules()
        with self.assertRaisesRegex(RuntimeError, "file set mismatch"):
            self._verify()

    def test_extra_generated_directory_is_rejected(self) -> None:
        self._thaw_modules()
        (self.modules / self.repo_relative / ("a" * 40)).mkdir()
        self._freeze_modules()
        with self.assertRaisesRegex(RuntimeError, "directory set mismatch"):
            self._verify()

    def test_generated_symlink_is_rejected(self) -> None:
        self._thaw_modules()
        target = self.modules / self.revision_relative / "utils.py"
        target.unlink()
        target.symlink_to(self.modules / self.revision_relative / "hf_model.py")
        self._freeze_modules()
        with self.assertRaisesRegex(RuntimeError, "not a real regular file"):
            self._verify()

    def test_writable_generated_cache_is_rejected(self) -> None:
        self._thaw_modules()
        with self.assertRaisesRegex(RuntimeError, "not frozen mode"):
            self._verify()

    def test_hub_snapshot_escape_is_rejected(self) -> None:
        destination = self.snapshot / "utils.py"
        destination.unlink()
        outside = self.repo_root / "outside"
        outside.write_bytes(self.source_payloads["utils.py"])
        destination.symlink_to("../../outside")
        with self.assertRaisesRegex(RuntimeError, "blobs namespace"):
            self._verify()

    def test_wrong_revision_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wrong remote revision"):
            self._verify(revision="a" * 40)

    def test_cwd_fallback_observation_is_rejected(self) -> None:
        fallback = self.base / "hf_model.py"
        fallback.write_bytes(self.source_payloads["hf_model.py"])
        with self.assertRaisesRegex(RuntimeError, "remote entry modules were not loaded"):
            self._write_observation(wrong_file=fallback)

    def test_observation_path_escape_is_rejected(self) -> None:
        value = json.loads(self.observation.read_text())
        row = next(row for row in value["loaded_modules"] if row["kind"] == "remote-source")
        row["file"] = str(self.base / "hf_model.py")
        body = dict(value)
        del body["manifest_sha256"]
        value["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        self.observation.write_bytes(_canonical(value))
        with self.assertRaisesRegex(RuntimeError, "file path mismatch"):
            self._verify()

    def test_observation_manifest_mismatch_is_rejected(self) -> None:
        value = json.loads(self.observation.read_text())
        value["cwd"] = "/changed"
        self.observation.write_bytes(_canonical(value))
        with self.assertRaisesRegex(RuntimeError, "execution observation manifest mismatch"):
            self._verify()

    def test_source_verification_manifest_mismatch_is_rejected(self) -> None:
        value = json.loads(self.source_evidence.read_text())
        value["remote_code"][0]["revision"] = "a" * 40
        self.source_evidence.write_bytes(_canonical(value))
        with self.assertRaisesRegex(RuntimeError, "Hub snapshot verification manifest mismatch"):
            self._verify()

    def test_non_finite_json_is_rejected(self) -> None:
        self.observation.write_bytes(b'{"manifest_sha256":NaN}\n')
        with self.assertRaisesRegex(RuntimeError, "non-finite JSON"):
            self._verify()

    def test_duplicate_json_key_is_rejected(self) -> None:
        self.observation.write_bytes(
            b'{"manifest_sha256":"' + b"0" * 64 + b'","manifest_sha256":"' + b"1" * 64 + b'"}\n'
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON object key"):
            self._verify()

    def test_missing_executed_entry_is_rejected(self) -> None:
        value = json.loads(self.observation.read_text())
        value["loaded_modules"] = []
        body = dict(value)
        del body["manifest_sha256"]
        value["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        self.observation.write_bytes(_canonical(value))
        with self.assertRaisesRegex(RuntimeError, "missing entry modules"):
            self._verify()

    def test_capture_rejects_modules_cache_after_cwd(self) -> None:
        module_name = self.prefix + ".hf_model"
        module = types.ModuleType(module_name)
        module.__file__ = str(
            (self.modules / self.revision_relative / "hf_model.py").resolve()
        )
        path = [str(self.base.resolve()), str(self.modules.resolve()), *sys.path]
        with mock.patch.dict(os.environ, self.environment, clear=False):
            with mock.patch.object(sys, "path", path):
                with mock.patch.dict(sys.modules, {module_name: module}, clear=False):
                    with self.assertRaisesRegex(RuntimeError, "first sys.path entry"):
                        VERIFIER.capture_execution_observation(
                            self.modules,
                            REPOSITORY,
                            REVISION,
                            self.closure,
                            self.entries,
                        )

    def test_loaded_module_from_old_revision_is_rejected(self) -> None:
        current_name = self.prefix + ".hf_model"
        current = types.ModuleType(current_name)
        current.__file__ = str(
            (self.modules / self.revision_relative / "hf_model.py").resolve()
        )
        old_revision = "a" * 40
        old_name = ".".join(self.repo_relative.parts) + f".{old_revision}.hf_model"
        old = types.ModuleType(old_name)
        old.__file__ = str(self.base / "old" / "hf_model.py")
        path = [str(self.modules.resolve()), *sys.path]
        with mock.patch.dict(os.environ, self.environment, clear=False):
            with mock.patch.object(sys, "path", path):
                with mock.patch.dict(
                    sys.modules, {current_name: current, old_name: old}, clear=False
                ):
                    observation = VERIFIER.capture_execution_observation(
                        self.modules,
                        REPOSITORY,
                        REVISION,
                        self.closure,
                        self.entries,
                    )
        self.observation.write_bytes(_canonical(observation))
        with self.assertRaisesRegex(RuntimeError, "unexpected namespace modules"):
            self._verify()

    def test_effective_bytecode_writes_are_rejected(self) -> None:
        with mock.patch.dict(os.environ, self.environment, clear=False):
            with mock.patch.object(sys, "dont_write_bytecode", False):
                with self.assertRaisesRegex(RuntimeError, "effective Python -B"):
                    VERIFIER.verify(
                        self.modules,
                        self.hub_cache,
                        self.source_evidence,
                        self.observation,
                        REPOSITORY,
                        REVISION,
                        VERIFIER.TRANSFORMERS_VERSION,
                    )

    def test_wrong_hf_modules_environment_is_rejected(self) -> None:
        wrong = self.base / "wrong-modules"
        wrong.mkdir()
        environment = dict(self.environment)
        environment["HF_MODULES_CACHE"] = str(wrong)
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "different generated cache root"):
                VERIFIER.verify(
                    self.modules,
                    self.hub_cache,
                    self.source_evidence,
                    self.observation,
                    REPOSITORY,
                    REVISION,
                    VERIFIER.TRANSFORMERS_VERSION,
                )

    def test_wrong_transformers_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported Transformers version"):
            self._verify(transformers_version="5.8.0")

    def test_shared_cas_preflight_avoids_second_blob_rehash(self) -> None:
        preflight = VERIFIER.build_cas_preflight(
            self.hub_cache,
            self.source_evidence,
            REPOSITORY,
            REVISION,
        )
        preflight_path = self.base / "cas-preflight.json"
        VERIFIER.write_canonical_noreplace(preflight_path, preflight)
        with mock.patch.object(
            VERIFIER,
            "_verify_remote_snapshot",
            side_effect=AssertionError("CAS must not be rehashed in a lane"),
        ):
            result = self._verify(cas_preflight_path=preflight_path)
        self.assertEqual(
            result["hub_cas_preflight_manifest_sha256"],
            preflight["manifest_sha256"],
        )

    def test_tampered_cas_preflight_is_rejected(self) -> None:
        preflight = VERIFIER.build_cas_preflight(
            self.hub_cache,
            self.source_evidence,
            REPOSITORY,
            REVISION,
        )
        preflight["sources"][0]["size"] += 1
        preflight_path = self.base / "cas-preflight.json"
        preflight_path.write_bytes(_canonical(preflight))
        with self.assertRaisesRegex(RuntimeError, "Hub CAS preflight manifest mismatch"):
            self._verify(cas_preflight_path=preflight_path)

    def test_output_is_canonical_read_only_and_no_replace(self) -> None:
        result = self._verify()
        output = self.base / "result.json"
        VERIFIER.write_canonical_noreplace(output, result)
        self.assertEqual(output.read_bytes(), _canonical(result))
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
        with self.assertRaises(FileExistsError):
            VERIFIER.write_canonical_noreplace(output, result)

    def test_output_broken_symlink_is_not_replaced(self) -> None:
        output = self.base / "result.json"
        output.symlink_to(self.base / "missing")
        with self.assertRaises(FileExistsError):
            VERIFIER.write_canonical_noreplace(output, {"ok": True})
        self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
