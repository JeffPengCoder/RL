#!/usr/bin/env python3
"""Bind a Transformers 5.8.1 dynamic-module cache to verified Hub CAS bytes.

This verifier is intentionally narrow.  It qualifies one *dedicated, fresh*
``HF_MODULES_CACHE`` after a remote-code CPU preflight and before GPU work.
For Transformers 5.8.1, remote code is copied to::

    HF_MODULES_CACHE/
      transformers_modules/<sanitized-owner>/<sanitized-repo>/<commit>/...

The copied file is reused when it already exists, so verifying only the Hub
snapshot is insufficient.  This module independently rehashes the Hub CAS,
requires an exact generated-cache tree, and joins it to a same-process
``sys.modules`` observation captured after import.

Production callers should materialize in a unique ``.partial`` with ``-B`` and
the exact ``HF_MODULES_CACHE`` first on ``sys.path``.  Capture the execution
observation in that process, freeze files 0444/directories 0555, attest, and
atomically publish without replacement.  A fresh process must load the same
published read-only tree, capture a final-path observation, and run this gate.
The cache is exclusive: unrelated local or remote dynamic modules are rejected.
Use a separate fresh cache for this gate.  ``build_cas_preflight`` can rehash a
sealed Hub CAS once for several isolated lanes; pass that immutable result as
``--cas-preflight`` to avoid repeated large-CAS reads.

The layout is derived from the official Transformers 5.8.1
``dynamic_module_utils.py`` implementation.  A real-container execution
observation is still required for qualification; synthetic unit fixtures are
not runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Iterable


FORMAT = "hf-modules-cache-verification-v1"
OBSERVATION_FORMAT = "hf-modules-execution-observation-v1"
CAS_PREFLIGHT_FORMAT = "hf-modules-cas-preflight-v1"
SOURCE_FORMAT = "hf-snapshot-root-mount-verification-v3"
LAYOUT = "transformers-5.8.1-remote-code"
TRANSFORMERS_VERSION = "5.8.1"
DYNAMIC_MODULE_ROOT = "transformers_modules"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REPO_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_SOURCE_BYTES = 32 * 1024 * 1024


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_regular_file(
    path: Path, label: str, *, maximum_bytes: int | None = None
) -> tuple[bytes, str, int, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            f"{label} is missing, inaccessible, or a symlink: {path}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise RuntimeError(
                f"{label} exceeds the supported size: {path}; bytes={before.st_size}"
            )
        payload = bytearray()
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    return bytes(payload), digest.hexdigest(), before.st_size, _identity(before)


def _stable_blob(
    path: Path,
) -> tuple[str, str, int, bytes | None, str, tuple[int, ...]]:
    name = path.name
    if HEX64.fullmatch(name):
        algorithm = "sha256"
        named_digest = hashlib.sha256()
        git_header = b""
    elif HEX40.fullmatch(name):
        algorithm = "git-blob-sha1"
        named_digest = hashlib.sha1(usedforsecurity=False)
        git_header = None
    else:
        raise RuntimeError(f"unsupported Hugging Face blob identity: {name}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Hub blob is not a regular file: {path}")
        if git_header is None:
            git_header = f"blob {before.st_size}\0".encode()
        named_digest.update(git_header)
        raw_digest = hashlib.sha256()
        capture = bytearray() if before.st_size <= MAX_SOURCE_BYTES else None
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            named_digest.update(chunk)
            raw_digest.update(chunk)
            if capture is not None:
                capture.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"Hub blob changed while being read: {path}")
    finally:
        os.close(descriptor)

    observed = named_digest.hexdigest()
    if observed != name:
        raise RuntimeError(
            f"Hub blob digest mismatch: {path}; expected={name}; observed={observed}"
        )
    return (
        algorithm,
        observed,
        before.st_size,
        bytes(capture) if capture is not None else None,
        raw_digest.hexdigest(),
        _identity(before),
    )


def _require_real_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is missing or inaccessible: {path}: {error}") from error
    if not resolved.is_dir():
        raise RuntimeError(f"{label} must be a directory: {path}")
    return resolved


def _strict_json(payload: bytes, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid {label} JSON: {error}") from error


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    payload, _, _, _ = _stable_regular_file(path, label, maximum_bytes=MAX_JSON_BYTES)
    value = _strict_json(payload, label)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _verify_self_manifest(value: dict[str, Any], label: str) -> str:
    observed = value.get("manifest_sha256")
    if not isinstance(observed, str) or not HEX64.fullmatch(observed):
        raise RuntimeError(f"{label} has no valid manifest_sha256")
    body = dict(value)
    del body["manifest_sha256"]
    expected = hashlib.sha256(_canonical_json(body)).hexdigest()
    if observed != expected:
        raise RuntimeError(
            f"{label} manifest mismatch: recorded={observed}; computed={expected}"
        )
    return observed


def _repo_cache_name(repository: str) -> str:
    return "models--" + repository.replace("/", "--")


def _sanitize_component(name: str) -> str:
    sanitized = name.replace(".", "_dot_").replace("-", "_hyphen_")
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if not sanitized or "/" in sanitized or "\\" in sanitized:
        raise RuntimeError(f"repository component cannot be sanitized safely: {name!r}")
    return sanitized


def _generated_repository_relative(repository: str) -> PurePosixPath:
    if not REPO_ID.fullmatch(repository):
        raise RuntimeError(f"invalid Hugging Face repository id: {repository!r}")
    owner, name = repository.split("/", 1)
    return PurePosixPath(
        DYNAMIC_MODULE_ROOT,
        _sanitize_component(owner),
        _sanitize_component(name),
    )


def _validate_source_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
        or not value.endswith(".py")
    ):
        raise RuntimeError(f"invalid {label}: {value!r}")
    if path.name == "__init__.py":
        raise RuntimeError(
            f"package __init__.py remote sources are unsupported by this exact layout: {value}"
        )
    return path


def _snapshot_entries(snapshot: Path) -> list[Path]:
    entries: list[Path] = []
    for directory, names, files in os.walk(snapshot, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"Hub snapshot directory is not a real directory: {child}")
        entries.extend(directory_path / name for name in files)
    return sorted(entries, key=lambda item: item.relative_to(snapshot).as_posix())


def _verify_remote_snapshot(
    cache_root: Path, repository: str, revision: str
) -> dict[str, Any]:
    root = _require_real_directory(cache_root, "Hub cache root")
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"remote-code revision must be a 40-character commit: {revision}")
    repo_root = _require_real_directory(
        root / _repo_cache_name(repository), "remote-code repository root"
    )
    blobs = _require_real_directory(repo_root / "blobs", "remote-code blob root")
    snapshot = _require_real_directory(
        repo_root / "snapshots" / revision, "remote-code snapshot"
    )

    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    raw_sha256: dict[str, str] = {}
    observed: list[tuple[Path, str, Path, tuple[int, ...]]] = []
    cached: dict[
        Path, tuple[str, str, int, bytes | None, str, tuple[int, ...]]
    ] = {}
    for entry in _snapshot_entries(snapshot):
        if not stat.S_ISLNK(entry.lstat().st_mode):
            raise RuntimeError(f"Hub snapshot leaf is not a symlink: {entry}")
        link_value = os.readlink(entry)
        if os.path.isabs(link_value):
            raise RuntimeError(f"Hub snapshot link is absolute: {entry} -> {link_value}")
        target = (entry.parent / link_value).resolve(strict=True)
        if target.parent != blobs or not (
            HEX40.fullmatch(target.name) or HEX64.fullmatch(target.name)
        ):
            raise RuntimeError(
                f"Hub snapshot link escapes the exact blobs namespace: {entry} -> {target}"
            )
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise RuntimeError(f"Hub snapshot target is not a real regular blob: {target}")
        if target not in cached:
            cached[target] = _stable_blob(target)
        algorithm, digest, size, payload, raw_digest, identity = cached[target]
        relative = entry.relative_to(snapshot).as_posix()
        if payload is not None:
            payloads[relative] = payload
        raw_sha256[relative] = raw_digest
        records.append(
            {
                "blob_algorithm": algorithm,
                "blob_digest": digest,
                "link": link_value,
                "path": relative,
                "size": size,
                "target_relative": target.relative_to(repo_root).as_posix(),
            }
        )
        observed.append((entry, link_value, target, identity))

    for entry, link_value, target, identity in observed:
        if os.readlink(entry) != link_value:
            raise RuntimeError(f"Hub snapshot link changed during verification: {entry}")
        if _identity(target.lstat()) != identity:
            raise RuntimeError(f"Hub blob identity changed after verification: {target}")
    final_entries = [
        entry.relative_to(snapshot).as_posix() for entry in _snapshot_entries(snapshot)
    ]
    if final_entries != [record["path"] for record in records]:
        raise RuntimeError("Hub snapshot entry set changed during verification")

    content = {"files": records, "revision": revision}
    return {
        "content_manifest_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        "files": records,
        "payloads": payloads,
        "raw_sha256": raw_sha256,
        "repo_root": str(repo_root),
    }


def _source_contract(
    verification: dict[str, Any],
    repository: str,
    revision: str,
    hub_cache_root: Path,
) -> dict[str, Any]:
    source_manifest = _verify_self_manifest(verification, "Hub snapshot verification")
    if verification.get("format") != SOURCE_FORMAT:
        raise RuntimeError(
            f"unsupported Hub snapshot verification format: {verification.get('format')!r}"
        )
    remote_code = verification.get("remote_code")
    if not isinstance(remote_code, list):
        raise RuntimeError("Hub snapshot verification has no remote_code list")
    rows = [
        row
        for row in remote_code
        if isinstance(row, dict) and row.get("repository") == repository
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Hub snapshot verification must contain exactly one row for {repository}"
        )
    row = rows[0]
    if row.get("revision") != revision:
        raise RuntimeError(
            f"Hub snapshot verification has wrong remote revision: {row.get('revision')!r}"
        )
    content_manifest = row.get("content_manifest_sha256")
    if not isinstance(content_manifest, str) or not HEX64.fullmatch(content_manifest):
        raise RuntimeError("Hub snapshot verification has no valid remote content manifest")

    closure = row.get("module_closure")
    entries = row.get("entry_modules")
    if (
        not isinstance(closure, list)
        or not closure
        or any(not isinstance(item, str) for item in closure)
        or closure != sorted(set(closure))
    ):
        raise RuntimeError("remote module_closure must be a non-empty sorted unique list")
    if (
        not isinstance(entries, list)
        or not entries
        or any(not isinstance(item, str) for item in entries)
        or entries != sorted(set(entries))
        or not set(entries).issubset(closure)
    ):
        raise RuntimeError("remote entry_modules must be a non-empty sorted subset of closure")
    closure_paths = [_validate_source_path(item, "remote module path") for item in closure]
    entry_paths = [_validate_source_path(item, "remote entry module path") for item in entries]

    offline = verification.get("offline_environment")
    expected_hub_root = _require_real_directory(hub_cache_root, "Hub cache root")
    if not isinstance(offline, dict):
        raise RuntimeError("Hub snapshot verification lacks required offline environment evidence")
    if offline.get("hf_hub_offline") != "1" or offline.get("transformers_offline") != "1":
        raise RuntimeError("Hub snapshot verification was not produced under exact offline flags")
    recorded_hub = offline.get("hf_hub_cache")
    if not isinstance(recorded_hub, str):
        raise RuntimeError("Hub snapshot verification lacks hf_hub_cache evidence")
    if _require_real_directory(Path(recorded_hub), "recorded Hub cache root") != expected_hub_root:
        raise RuntimeError("Hub snapshot verification selects a different Hub cache root")

    return {
        "closure_paths": [path.as_posix() for path in closure_paths],
        "entry_paths": [path.as_posix() for path in entry_paths],
        "hub_cache_root": str(expected_hub_root),
        "remote_content_manifest_sha256": content_manifest,
        "source_manifest_sha256": source_manifest,
    }


def _source_authority(
    verification: dict[str, Any],
    repository: str,
    revision: str,
    hub_cache_root: Path,
) -> dict[str, Any]:
    contract = _source_contract(
        verification, repository, revision, hub_cache_root
    )
    expected_hub_root = Path(contract["hub_cache_root"])
    snapshot = _verify_remote_snapshot(expected_hub_root, repository, revision)
    content_manifest = contract["remote_content_manifest_sha256"]
    if snapshot["content_manifest_sha256"] != content_manifest:
        raise RuntimeError(
            "remote Hub snapshot changed after source verification: "
            f"expected={content_manifest}; observed={snapshot['content_manifest_sha256']}"
        )
    by_name = {record["path"]: record for record in snapshot["files"]}
    sources: list[dict[str, Any]] = []
    for relative in contract["closure_paths"]:
        record = by_name.get(relative)
        payload = snapshot["payloads"].get(relative)
        if record is None or payload is None:
            raise RuntimeError(f"remote closure source is absent or too large: {relative}")
        sources.append(
            {
                "blob_algorithm": record["blob_algorithm"],
                "blob_digest": record["blob_digest"],
                "path": relative,
                "payload": payload,
                "raw_sha256": snapshot["raw_sha256"][relative],
                "size": record["size"],
            }
        )
    return {
        "entry_paths": contract["entry_paths"],
        "remote_content_manifest_sha256": content_manifest,
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "sources": sources,
    }


def build_cas_preflight(
    hub_cache_root: Path,
    snapshot_verification_path: Path,
    repository: str,
    revision: str,
) -> dict[str, Any]:
    """Rehash the sealed Hub CAS once and return a shareable immutable authority."""
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"revision must be a 40-character commit: {revision}")
    hub_root = _require_real_directory(hub_cache_root, "Hub cache root")
    verification = _load_json_file(
        snapshot_verification_path, "Hub snapshot verification"
    )
    source = _source_authority(verification, repository, revision, hub_root)
    body = {
        "entry_paths": source["entry_paths"],
        "format": CAS_PREFLIGHT_FORMAT,
        "hub_cache_root": str(hub_root),
        "layout": LAYOUT,
        "remote_content_manifest_sha256": source[
            "remote_content_manifest_sha256"
        ],
        "repository": repository,
        "revision": revision,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "sources": [
            {
                "blob_algorithm": row["blob_algorithm"],
                "blob_digest": row["blob_digest"],
                "path": row["path"],
                "raw_sha256": row["raw_sha256"],
                "size": row["size"],
            }
            for row in source["sources"]
        ],
    }
    body["manifest_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _source_from_preflight(
    verification: dict[str, Any],
    preflight: dict[str, Any],
    repository: str,
    revision: str,
    hub_cache_root: Path,
) -> dict[str, Any]:
    preflight_manifest = _verify_self_manifest(preflight, "Hub CAS preflight")
    contract = _source_contract(
        verification, repository, revision, hub_cache_root
    )
    expected_scalars = {
        "format": CAS_PREFLIGHT_FORMAT,
        "hub_cache_root": contract["hub_cache_root"],
        "layout": LAYOUT,
        "remote_content_manifest_sha256": contract[
            "remote_content_manifest_sha256"
        ],
        "repository": repository,
        "revision": revision,
        "source_manifest_sha256": contract["source_manifest_sha256"],
    }
    for key, expected in expected_scalars.items():
        if preflight.get(key) != expected:
            raise RuntimeError(
                f"Hub CAS preflight has wrong {key}: "
                f"{preflight.get(key)!r}; expected={expected!r}"
            )
    if preflight.get("entry_paths") != contract["entry_paths"]:
        raise RuntimeError("Hub CAS preflight entry module set differs from source evidence")
    sources = preflight.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("Hub CAS preflight has no sources list")
    expected_keys = {
        "blob_algorithm",
        "blob_digest",
        "path",
        "raw_sha256",
        "size",
    }
    paths: list[str] = []
    normalized: list[dict[str, Any]] = []
    for row in sources:
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise RuntimeError("Hub CAS preflight source row has an invalid schema")
        path = row.get("path")
        algorithm = row.get("blob_algorithm")
        digest = row.get("blob_digest")
        raw_digest = row.get("raw_sha256")
        size = row.get("size")
        if not isinstance(path, str):
            raise RuntimeError("Hub CAS preflight source path is not a string")
        _validate_source_path(path, "Hub CAS preflight source path")
        if algorithm == "git-blob-sha1":
            valid_named_digest = isinstance(digest, str) and bool(HEX40.fullmatch(digest))
        elif algorithm == "sha256":
            valid_named_digest = isinstance(digest, str) and bool(HEX64.fullmatch(digest))
        else:
            valid_named_digest = False
        if (
            not valid_named_digest
            or not isinstance(raw_digest, str)
            or not HEX64.fullmatch(raw_digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_SOURCE_BYTES
        ):
            raise RuntimeError(f"Hub CAS preflight source metadata is invalid: {path}")
        paths.append(path)
        normalized.append(dict(row))
    if paths != contract["closure_paths"]:
        raise RuntimeError("Hub CAS preflight closure differs from source evidence")
    return {
        "cas_preflight_manifest_sha256": preflight_manifest,
        "entry_paths": contract["entry_paths"],
        "remote_content_manifest_sha256": contract[
            "remote_content_manifest_sha256"
        ],
        "source_manifest_sha256": contract["source_manifest_sha256"],
        "sources": normalized,
    }


def _module_name(prefix: str, source_relative: str) -> str:
    path = PurePosixPath(source_relative)
    return prefix + "." + ".".join(path.with_suffix("").parts)


def _expected_generated_tree(
    repository: str, revision: str, sources: Iterable[dict[str, Any]]
) -> tuple[dict[str, tuple[str, dict[str, Any] | None]], set[str], str]:
    repo_relative = _generated_repository_relative(repository)
    revision_relative = repo_relative / revision
    prefix = ".".join(revision_relative.parts)
    files: dict[str, tuple[str, dict[str, Any] | None]] = {"__init__.py": ("init", None)}
    directories: set[str] = {""}

    def add_directory(path: PurePosixPath) -> None:
        parts = path.parts
        for index in range(1, len(parts) + 1):
            relative = PurePosixPath(*parts[:index]).as_posix()
            directories.add(relative)
            init = PurePosixPath(relative, "__init__.py").as_posix()
            files.setdefault(init, ("init", None))

    add_directory(revision_relative)
    for source in sources:
        source_path = _validate_source_path(source["path"], "source authority path")
        parent = revision_relative / source_path.parent
        if source_path.parent != PurePosixPath("."):
            add_directory(parent)
        destination = (revision_relative / source_path).as_posix()
        if destination in files:
            raise RuntimeError(
                f"generated module path collides with structural file: {destination}"
            )
        files[destination] = ("source", source)
    return files, directories, prefix


def _enumerate_generated_tree(root: Path) -> tuple[set[str], set[str]]:
    directories: set[str] = {""}
    files: set[str] = set()
    for directory, names, leaves in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"generated cache directory is not a real directory: {child}")
            directories.add(child.relative_to(root).as_posix())
        for name in leaves:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"generated cache leaf is not a real regular file: {child}")
            files.add(child.relative_to(root).as_posix())
    return directories, files


def _verify_generated_tree(
    modules_cache_root: Path,
    expected_files: dict[str, tuple[str, dict[str, Any] | None]],
    expected_directories: set[str],
) -> list[dict[str, Any]]:
    root = _require_real_directory(modules_cache_root, "HF modules cache root")
    observed_directories, observed_files = _enumerate_generated_tree(root)
    if observed_directories != expected_directories:
        raise RuntimeError(
            "generated cache directory set mismatch: "
            f"expected={sorted(expected_directories)}; observed={sorted(observed_directories)}"
        )
    if observed_files != set(expected_files):
        raise RuntimeError(
            "generated cache file set mismatch: "
            f"expected={sorted(expected_files)}; observed={sorted(observed_files)}"
        )

    for relative in sorted(observed_directories):
        path = root if not relative else root / PurePosixPath(relative)
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != 0o555:
            raise RuntimeError(
                f"generated cache directory is not frozen mode 0555: {path}; "
                f"mode={mode:04o}"
            )

    records: list[dict[str, Any]] = []
    identities: list[tuple[Path, tuple[int, ...]]] = []
    for relative in sorted(expected_files):
        kind, source = expected_files[relative]
        path = root / PurePosixPath(relative)
        payload, digest, size, identity = _stable_regular_file(
            path, "generated module file", maximum_bytes=MAX_SOURCE_BYTES
        )
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode != 0o444:
            raise RuntimeError(
                f"generated cache file is not frozen mode 0444: {path}; mode={mode:04o}"
            )
        identities.append((path, identity))
        if kind == "init":
            if payload:
                raise RuntimeError(f"generated structural __init__.py is not empty: {path}")
            records.append(
                {"kind": "structural-init", "path": relative, "sha256": digest, "size": 0}
            )
            continue
        assert source is not None
        if (
            size != source["size"]
            or digest != source["raw_sha256"]
            or ("payload" in source and payload != source["payload"])
        ):
            raise RuntimeError(
                f"generated module bytes do not match verified Hub CAS: {relative}"
            )
        records.append(
            {
                "blob_algorithm": source["blob_algorithm"],
                "blob_digest": source["blob_digest"],
                "kind": "remote-source",
                "path": relative,
                "sha256": digest,
                "size": size,
                "source_relative": source["path"],
            }
        )

    final_directories, final_files = _enumerate_generated_tree(root)
    if final_directories != observed_directories or final_files != observed_files:
        raise RuntimeError("generated cache entry set changed during verification")
    for path, identity in identities:
        if _identity(path.lstat()) != identity:
            raise RuntimeError(f"generated cache file changed after verification: {path}")
    return records


def _runtime_environment(
    modules_cache_root: Path, hub_cache_root: Path
) -> dict[str, str]:
    expected_modules = _require_real_directory(modules_cache_root, "HF modules cache root")
    expected_hub = _require_real_directory(hub_cache_root, "Hub cache root")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1")
    configured_modules = os.environ.get("HF_MODULES_CACHE")
    configured_hub = os.environ.get("HF_HUB_CACHE")
    if not configured_modules or not configured_hub:
        raise RuntimeError("HF_MODULES_CACHE and HF_HUB_CACHE must both be set")
    if _require_real_directory(Path(configured_modules), "HF_MODULES_CACHE") != expected_modules:
        raise RuntimeError("HF_MODULES_CACHE selects a different generated cache root")
    if _require_real_directory(Path(configured_hub), "HF_HUB_CACHE") != expected_hub:
        raise RuntimeError("HF_HUB_CACHE selects a different Hub cache root")
    if not sys.dont_write_bytecode:
        raise RuntimeError("verifier must run with effective Python -B / dont_write_bytecode")
    return {
        "hf_hub_cache": str(expected_hub),
        "hf_hub_offline": "1",
        "hf_modules_cache": str(expected_modules),
        "transformers_offline": "1",
    }


def capture_execution_observation(
    modules_cache_root: Path,
    repository: str,
    revision: str,
    module_closure: Iterable[str],
    entry_modules: Iterable[str],
    *,
    transformers_version: str = TRANSFORMERS_VERSION,
) -> dict[str, Any]:
    """Capture loaded-module provenance inside the process that imported code."""
    if transformers_version != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "unsupported Transformers version: "
            f"{transformers_version}; expected={TRANSFORMERS_VERSION}"
        )
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"revision must be a 40-character commit: {revision}")
    root = _require_real_directory(modules_cache_root, "HF modules cache root")
    closure = sorted(set(module_closure))
    entries = sorted(set(entry_modules))
    if not closure or not entries or not set(entries).issubset(closure):
        raise RuntimeError("entry modules must be a non-empty subset of module closure")
    paths = [_validate_source_path(item, "captured module path") for item in closure]
    entry_paths = [_validate_source_path(item, "captured entry module path") for item in entries]
    repo_relative = _generated_repository_relative(repository)
    revision_relative = repo_relative / revision
    repository_prefix = ".".join(repo_relative.parts)
    prefix = ".".join(revision_relative.parts)
    expected: dict[str, tuple[str, Path]] = {}
    package_names: dict[str, Path] = {prefix: root / revision_relative / "__init__.py"}
    for path in paths:
        name = _module_name(prefix, path.as_posix())
        expected[name] = (path.as_posix(), root / revision_relative / path)
        parent = path.parent
        while parent != PurePosixPath("."):
            package_name = prefix + "." + ".".join(parent.parts)
            package_names[package_name] = root / revision_relative / parent / "__init__.py"
            parent = parent.parent

    observed_sys_path = [str(item) for item in sys.path]
    if not observed_sys_path or observed_sys_path[0] != str(root):
        raise RuntimeError(
            "HF_MODULES_CACHE must be the exact first sys.path entry in the import process"
        )
    if not sys.dont_write_bytecode:
        raise RuntimeError("import process must use effective Python -B / dont_write_bytecode")

    loaded: list[dict[str, Any]] = []
    unexpected: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            continue
        if name in expected:
            source_relative, canonical = expected[name]
            if filename != str(canonical):
                unexpected.append({"file": filename, "module_name": name})
                continue
            payload, digest, size, _ = _stable_regular_file(
                canonical, "executed generated module", maximum_bytes=MAX_SOURCE_BYTES
            )
            del payload
            loaded.append(
                {
                    "file": str(canonical),
                    "kind": "remote-source",
                    "module_name": name,
                    "sha256": digest,
                    "size": size,
                    "source_relative": source_relative,
                }
            )
        elif name in package_names:
            canonical = package_names[name]
            if filename != str(canonical):
                unexpected.append({"file": filename, "module_name": name})
                continue
            payload, digest, size, _ = _stable_regular_file(
                canonical, "executed generated package", maximum_bytes=MAX_SOURCE_BYTES
            )
            if payload:
                raise RuntimeError(
                    f"executed generated package initializer is not empty: {canonical}"
                )
            loaded.append(
                {
                    "file": str(canonical),
                    "kind": "structural-init",
                    "module_name": name,
                    "sha256": digest,
                    "size": size,
                }
            )
        elif name.startswith(repository_prefix + "."):
            unexpected.append({"file": filename, "module_name": name})

    shadow_modules: list[dict[str, str]] = []
    for path in paths:
        bare_name = ".".join(path.with_suffix("").parts)
        module = sys.modules.get(bare_name)
        filename = getattr(module, "__file__", None) if isinstance(module, ModuleType) else None
        if isinstance(filename, str):
            shadow_modules.append({"file": filename, "module_name": bare_name})

    loaded_names = {row["module_name"] for row in loaded}
    required_names = {_module_name(prefix, path.as_posix()) for path in entry_paths}
    if not required_names.issubset(loaded_names):
        raise RuntimeError(
            "remote entry modules were not loaded: "
            f"{sorted(required_names - loaded_names)}; unexpected={unexpected}"
        )
    body = {
        "cwd": str(Path.cwd().resolve()),
        "environment": {
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE", ""),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
            "hf_modules_cache": os.environ.get("HF_MODULES_CACHE", ""),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        },
        "format": OBSERVATION_FORMAT,
        "layout": LAYOUT,
        "loaded_modules": loaded,
        "modules_cache_root": str(root),
        "namespace": prefix,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "python_executable": str(Path(sys.executable).resolve()),
        "repository": repository,
        "revision": revision,
        "shadow_modules": shadow_modules,
        "sys_path": observed_sys_path,
        "transformers_version": transformers_version,
        "unexpected_namespace_modules": unexpected,
    }
    body["manifest_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _verify_execution_observation(
    observation: dict[str, Any],
    modules_cache_root: Path,
    hub_cache_root: Path,
    repository: str,
    revision: str,
    prefix: str,
    sources: list[dict[str, Any]],
    entry_paths: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    manifest = _verify_self_manifest(observation, "execution observation")
    expected_scalars = {
        "format": OBSERVATION_FORMAT,
        "layout": LAYOUT,
        "repository": repository,
        "revision": revision,
        "transformers_version": TRANSFORMERS_VERSION,
        "namespace": prefix,
        "modules_cache_root": str(modules_cache_root),
        "python_dont_write_bytecode": True,
    }
    for key, expected in expected_scalars.items():
        if observation.get(key) != expected:
            raise RuntimeError(
                f"execution observation has wrong {key}: "
                f"{observation.get(key)!r}; expected={expected!r}"
            )
    sys_path = observation.get("sys_path")
    if not isinstance(sys_path, list) or not sys_path or sys_path[0] != str(modules_cache_root):
        raise RuntimeError("execution observation does not put HF_MODULES_CACHE first on sys.path")
    environment = observation.get("environment")
    if not isinstance(environment, dict):
        raise RuntimeError("execution observation has no environment object")
    expected_environment = {
        "hf_hub_cache": str(hub_cache_root),
        "hf_hub_offline": "1",
        "hf_modules_cache": str(modules_cache_root),
        "transformers_offline": "1",
    }
    if environment != expected_environment:
        raise RuntimeError(
            f"execution observation environment mismatch: {environment!r}"
        )
    if observation.get("shadow_modules") != []:
        raise RuntimeError("execution observation found cwd/bare-name shadow modules")
    if observation.get("unexpected_namespace_modules") != []:
        raise RuntimeError("execution observation found unexpected namespace modules")

    by_source = {source["path"]: source for source in sources}
    expected_names = {
        _module_name(prefix, source["path"]): source for source in sources
    }
    revision_relative = _generated_repository_relative(repository) / revision
    expected_packages: dict[str, Path] = {
        prefix: modules_cache_root / revision_relative / "__init__.py"
    }
    for source in sources:
        parent = PurePosixPath(source["path"]).parent
        while parent != PurePosixPath("."):
            package_name = prefix + "." + ".".join(parent.parts)
            expected_packages[package_name] = (
                modules_cache_root / revision_relative / parent / "__init__.py"
            )
            parent = parent.parent
    loaded = observation.get("loaded_modules")
    if not isinstance(loaded, list):
        raise RuntimeError("execution observation has no loaded_modules list")
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_files: set[str] = set()
    for row in loaded:
        if not isinstance(row, dict):
            raise RuntimeError("execution observation module row is not an object")
        name = row.get("module_name")
        filename = row.get("file")
        kind = row.get("kind")
        if not isinstance(name, str) or not isinstance(filename, str):
            raise RuntimeError("execution observation module row lacks name or file")
        if name in seen_names or filename in seen_files:
            raise RuntimeError("execution observation contains duplicate module or file")
        seen_names.add(name)
        seen_files.add(filename)
        if kind == "structural-init":
            # Package rows are useful provenance but are not trainable source.
            path = expected_packages.get(name)
            if path is None or filename != str(path):
                raise RuntimeError(
                    f"execution observation structural initializer path mismatch: {name}"
                )
            payload, digest, size, _ = _stable_regular_file(
                path, "observed structural initializer", maximum_bytes=MAX_SOURCE_BYTES
            )
            if payload or digest != row.get("sha256") or size != row.get("size"):
                raise RuntimeError("execution observation structural initializer changed")
            records.append(dict(row))
            continue
        source = expected_names.get(name)
        if kind != "remote-source" or source is None:
            raise RuntimeError(f"execution observation has unexpected loaded module: {name}")
        source_relative = source["path"]
        expected_file = modules_cache_root / revision_relative / source_relative
        if filename != str(expected_file) or row.get("source_relative") != source_relative:
            raise RuntimeError(f"execution observation file path mismatch for {name}")
        payload, digest, size, _ = _stable_regular_file(
            expected_file, "observed executed module", maximum_bytes=MAX_SOURCE_BYTES
        )
        if (
            ("payload" in source and payload != source["payload"])
            or digest != source["raw_sha256"]
            or size != source["size"]
            or row.get("sha256") != digest
            or row.get("size") != size
        ):
            raise RuntimeError(f"executed module no longer matches Hub CAS: {name}")
        records.append(dict(row))

    required_names = {_module_name(prefix, path) for path in entry_paths}
    if not required_names.issubset(seen_names):
        raise RuntimeError(
            f"execution observation is missing entry modules: {sorted(required_names - seen_names)}"
        )
    # No source row can name a source that was not committed by the Hub authority.
    if any(
        row.get("kind") == "remote-source"
        and row.get("source_relative") not in by_source
        for row in records
    ):
        raise RuntimeError("execution observation contains an unbound source row")
    return manifest, records


def verify(
    modules_cache_root: Path,
    hub_cache_root: Path,
    snapshot_verification_path: Path,
    execution_observation_path: Path,
    repository: str,
    revision: str,
    transformers_version: str,
    cas_preflight_path: Path | None = None,
) -> dict[str, Any]:
    if transformers_version != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "unsupported Transformers version: "
            f"{transformers_version}; expected={TRANSFORMERS_VERSION}"
        )
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"revision must be a 40-character commit: {revision}")
    modules_root = _require_real_directory(modules_cache_root, "HF modules cache root")
    hub_root = _require_real_directory(hub_cache_root, "Hub cache root")
    environment = _runtime_environment(modules_root, hub_root)

    source_verification = _load_json_file(
        snapshot_verification_path, "Hub snapshot verification"
    )
    if cas_preflight_path is None:
        source = _source_authority(
            source_verification, repository, revision, hub_root
        )
    else:
        preflight = _load_json_file(cas_preflight_path, "Hub CAS preflight")
        source = _source_from_preflight(
            source_verification, preflight, repository, revision, hub_root
        )
    expected_files, expected_directories, prefix = _expected_generated_tree(
        repository, revision, source["sources"]
    )
    generated_records = _verify_generated_tree(
        modules_root, expected_files, expected_directories
    )
    observation = _load_json_file(
        execution_observation_path, "execution observation"
    )
    observation_manifest, executed_records = _verify_execution_observation(
        observation,
        modules_root,
        hub_root,
        repository,
        revision,
        prefix,
        source["sources"],
        source["entry_paths"],
    )
    final_generated_records = _verify_generated_tree(
        modules_root, expected_files, expected_directories
    )
    if final_generated_records != generated_records:
        raise RuntimeError("generated cache changed after execution-observation verification")

    result = {
        "environment": environment,
        "executed_modules": executed_records,
        "execution_observation_manifest_sha256": observation_manifest,
        "format": FORMAT,
        "generated_files": generated_records,
        "hub_cache_root": str(hub_root),
        "hub_remote_content_manifest_sha256": source[
            "remote_content_manifest_sha256"
        ],
        "hub_snapshot_verification_manifest_sha256": source[
            "source_manifest_sha256"
        ],
        "layout": LAYOUT,
        "modules_cache_root": str(modules_root),
        "namespace": prefix,
        "remote_entry_modules": source["entry_paths"],
        "remote_source_count": len(source["sources"]),
        "repository": repository,
        "revision": revision,
        "runtime_observation": "same-process-sys.modules-observation-verified",
        "transformers_version": transformers_version,
    }
    if "cas_preflight_manifest_sha256" in source:
        result["hub_cas_preflight_manifest_sha256"] = source[
            "cas_preflight_manifest_sha256"
        ]
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def write_canonical_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"output parent is not a real directory: {path.parent}")
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    committed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        os.link(partial, path, follow_symlinks=False)
        committed = True
        try:
            partial.unlink()
        except OSError:
            pass
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if not committed:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules-cache-root", type=Path, required=True)
    parser.add_argument("--hub-cache-root", type=Path, required=True)
    parser.add_argument("--snapshot-verification", type=Path, required=True)
    parser.add_argument("--cas-preflight", type=Path)
    parser.add_argument("--execution-observation", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.modules_cache_root,
        args.hub_cache_root,
        args.snapshot_verification,
        args.execution_observation,
        args.repository,
        args.revision,
        args.transformers_version,
        args.cas_preflight,
    )
    write_canonical_noreplace(args.output, result)
    print(
        "HF_MODULES_CACHE_VERIFIED|"
        f"repository={result['repository']}|revision={result['revision']}|"
        f"sources={result['remote_source_count']}|"
        f"executed={len(result['executed_modules'])}|"
        f"manifest={result['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
