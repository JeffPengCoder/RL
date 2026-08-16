#!/usr/bin/env python3
"""Run a parallel, CPU-only Hugging Face dynamic-code qualification bundle.

One parent process rehashes the sealed Hub CAS and publishes an immutable CAS
preflight.  Four subprocess lanes then use disjoint writable roots:

* ``remote-positive`` materializes C-RADIO remote code with ``AutoConfig``;
* ``processor-positive`` materializes the Nano-Omni ``AutoProcessor``;
* ``byte-tamper`` proves a changed attempt-copy source is rejected; and
* ``stale-replay`` proves an additional old revision is rejected.

Every positive cache is first built in a unique ``.partial`` directory,
attested, frozen to files 0444/directories 0555, and atomically published
without replacing an existing name.  A fresh process then loads the same
published read-only tree and proves that neither Transformers nor import
machinery writes to it.  Negative mutations operate only on per-lane copies;
the sealed Hub CAS is never modified.

Evidence includes safe argv, offline/cache authorities, secret-name presence
(never secret values or hashes), mountinfo, refs/revision, import events,
cache manifests, loaded ``__file__``/SHA provenance, traceback, and canonical
no-replace JSON.  Child stdout/stderr bytes are not persisted; only their size
and digest are recorded.  This stdlib runner must execute in the qualified
Linux container with the already-installed Transformers 5.8.1 runtime.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any


BUNDLE_FORMAT = "hf-dynamic-code-cpu-lane-bundle-v2"
LANE_FORMAT = "hf-dynamic-code-cpu-lane-result-v2"
CONSUMER_FORMAT = "hf-dynamic-code-readonly-consumer-v1"
PUBLICATION_FORMAT = "hf-modules-cache-publication-v1"
TRANSFORMERS_VERSION = "5.8.1"
LANES = (
    "remote-positive",
    "processor-positive",
    "byte-tamper",
    "stale-replay",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
SENSITIVE_ENV = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|CREDENTIAL|AUTH|COOKIE)(?:_|$)",
    re.IGNORECASE,
)
MAX_CAPTURE_BYTES = 32 * 1024 * 1024
MAX_MOUNTINFO_BYTES = 16 * 1024 * 1024
MAX_IMPORT_EVENTS = 20_000
SAFE_ENV_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_MODULES_CACHE",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "PYTHONSAFEPATH",
    "TRANSFORMERS_OFFLINE",
    "XDG_CACHE_HOME",
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_file(
    path: Path,
    label: str,
    *,
    maximum_bytes: int | None = MAX_CAPTURE_BYTES,
    capture: bool = True,
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
                f"{label} exceeds capture limit: {path}; bytes={before.st_size}"
            )
        payload = bytearray()
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                raise RuntimeError(
                    f"{label} exceeds capture limit while reading: {path}; bytes>{maximum_bytes}"
                )
            digest.update(chunk)
            if capture:
                payload.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    return bytes(payload), digest.hexdigest(), total, _identity(before)


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


def _future_path(path: Path, label: str) -> Path:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{label} must not already exist: {path}")
    parent = _require_real_directory(path.parent, f"{label} parent")
    return parent / path.name


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
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
        value = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid {label} JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _verify_self_manifest(value: dict[str, Any], label: str) -> str:
    observed = value.get("manifest_sha256")
    if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{64}", observed):
        raise RuntimeError(f"{label} has no valid manifest_sha256")
    body = dict(value)
    del body["manifest_sha256"]
    expected = _sha256(_canonical_json(body))
    if observed != expected:
        raise RuntimeError(
            f"{label} manifest mismatch: recorded={observed}; computed={expected}"
        )
    return observed


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload, _, _, _ = _stable_file(path, label)
    value = _strict_json(payload, label)
    _verify_self_manifest(value, label)
    return value


def _evidence_reference(
    path: Path,
    value: dict[str, Any],
    invocation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _, digest, size, _ = _stable_file(
        path, "referenced evidence", maximum_bytes=None, capture=False
    )
    reference: dict[str, Any] = {
        "format": value.get("format"),
        "manifest_sha256": value.get("manifest_sha256"),
        "path": str(path.resolve(strict=True)),
        "sha256": digest,
        "size": size,
        "status": value.get("status"),
    }
    if invocation is not None:
        reference["invocation"] = invocation
    return reference


def _require_consumer_publication_join(
    publication: dict[str, Any], consumer: dict[str, Any]
) -> None:
    before = consumer.get("cache_before")
    after = consumer.get("cache_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise RuntimeError("readonly consumer lacks before/after cache manifests")
    expected_identity = publication.get("destination_root_identity")
    observed_identity = before.get("root_identity")
    if (
        not isinstance(expected_identity, list)
        or not isinstance(observed_identity, list)
        or len(expected_identity) < 2
        or len(observed_identity) < 2
        or observed_identity[:2] != expected_identity[:2]
    ):
        raise RuntimeError("readonly consumer did not open the published root inode")
    expected_content = publication.get("content_manifest_sha256")
    if (
        before.get("content_manifest_sha256") != expected_content
        or after.get("content_manifest_sha256") != expected_content
        or before.get("entries") != after.get("entries")
    ):
        raise RuntimeError("readonly consumer did not preserve the published cache tree")


def _write_bytes_noreplace(path: Path, payload: bytes) -> None:
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


def _write_json_noreplace(path: Path, value: dict[str, Any]) -> None:
    _write_bytes_noreplace(path, _canonical_json(value))


def _environment_authority(environment: dict[str, str]) -> dict[str, Any]:
    return {
        "authorities": {
            name.lower(): environment.get(name, "") for name in SAFE_ENV_NAMES
        },
        "secret_presence": sorted(
            name
            for name, value in environment.items()
            if value and SENSITIVE_ENV.search(name)
        ),
    }


def _unescape_mount_field(value: str) -> str:
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _parse_mountinfo(payload: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in payload.decode("utf-8", "strict").splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as error:
            raise RuntimeError(f"malformed mountinfo line: {line!r}") from error
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeError(f"short mountinfo line: {line!r}")
        records.append(
            {
                "filesystem": fields[separator + 1],
                "mount_options": sorted(fields[5].split(",")),
                "mount_point": _unescape_mount_field(fields[4]),
                "root": _unescape_mount_field(fields[3]),
                "source": _unescape_mount_field(fields[separator + 2]),
                "super_options": sorted(fields[separator + 3].split(",")),
            }
        )
    if not records:
        raise RuntimeError("Linux mountinfo contains no records")
    return records


def _mountinfo_evidence() -> dict[str, Any]:
    path = Path("/proc/self/mountinfo")
    payload, digest, size, identity = _stable_file(
        path, "Linux mountinfo", maximum_bytes=MAX_MOUNTINFO_BYTES
    )
    return {
        "identity": list(identity),
        "records": _parse_mountinfo(payload),
        "sha256": digest,
        "size": size,
        "source": str(path),
    }


def _mount_for_path(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        mount = Path(record["mount_point"])
        if resolved == mount or _inside(resolved, mount):
            matches.append((len(mount.parts), record))
    if not matches:
        raise RuntimeError(f"no mountinfo record covers path: {resolved}")
    return max(matches, key=lambda item: item[0])[1]


def _mode_tree_read_only(root: Path) -> tuple[bool, list[str]]:
    writable: list[str] = []
    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for path in [directory_path, *(directory_path / name for name in names + files)]:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                writable.append(str(path))
    return not writable, writable[:100]


def _read_only_authority(
    path: Path, mountinfo: dict[str, Any], label: str
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    mount = _mount_for_path(resolved, mountinfo["records"])
    mount_read_only = "ro" in mount["mount_options"] or "ro" in mount["super_options"]
    if resolved.is_dir():
        mode_read_only, writable = _mode_tree_read_only(resolved)
    else:
        metadata = resolved.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"shared authority is not a real regular file: {resolved}")
        mode_read_only = not bool(stat.S_IMODE(metadata.st_mode) & 0o222)
        writable = [] if mode_read_only else [str(resolved)]
    if not mount_read_only and not mode_read_only:
        raise RuntimeError(
            f"shared authority is writable by mount and mode: {label}: {resolved}"
        )
    return {
        "label": label,
        "mode_read_only": mode_read_only,
        "mount": mount,
        "mount_read_only": mount_read_only,
        "path": str(resolved),
        "writable_examples": writable,
    }


def _tree_manifest(root: Path) -> dict[str, Any]:
    resolved = _require_real_directory(root, "cache manifest root")
    records: list[dict[str, Any]] = []
    for directory, names, files in os.walk(resolved, followlinks=False):
        directory_path = Path(directory)
        for name in sorted(names + files):
            path = directory_path / name
            relative = path.relative_to(resolved).as_posix()
            metadata = path.lstat()
            base: dict[str, Any] = {
                "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
                "path": relative,
            }
            if stat.S_ISLNK(metadata.st_mode):
                base.update({"kind": "symlink", "target": os.readlink(path)})
            elif stat.S_ISDIR(metadata.st_mode):
                base["kind"] = "directory"
            elif stat.S_ISREG(metadata.st_mode):
                _, digest, size, _ = _stable_file(
                    path,
                    "cache manifest file",
                    maximum_bytes=None,
                    capture=False,
                )
                base.update({"kind": "file", "sha256": digest, "size": size})
            else:
                base.update({"kind": "other", "type_bits": metadata.st_mode})
            records.append(base)
    content_manifest = _sha256(_canonical_json({"entries": records}))
    body = {
        "content_manifest_sha256": content_manifest,
        "entries": records,
        "root": str(resolved),
        "root_identity": list(_identity(resolved.lstat())),
    }
    body["manifest_sha256"] = _sha256(_canonical_json(body))
    return body


def _require_frozen_manifest(manifest: dict[str, Any]) -> None:
    for row in manifest["entries"]:
        if row["kind"] == "directory" and row["mode"] != "0555":
            raise RuntimeError(f"cache directory is not frozen 0555: {row['path']}")
        if row["kind"] == "file" and row["mode"] != "0444":
            raise RuntimeError(f"cache file is not frozen 0444: {row['path']}")
        if row["kind"] not in {"directory", "file"}:
            raise RuntimeError(
                f"cache contains unsupported {row['kind']} entry: {row['path']}"
            )


def _freeze_tree(root: Path) -> dict[str, Any]:
    resolved = _require_real_directory(root, "partial modules cache")
    directories: list[Path] = []
    files: list[Path] = []
    for directory, names, leaves in os.walk(resolved, followlinks=False):
        directory_path = Path(directory)
        directories.append(directory_path)
        for name in names:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"modules cache has a non-directory child: {child}")
        for name in leaves:
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"modules cache has a non-regular leaf: {child}")
            files.append(child)
    for path in sorted(files):
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    manifest = _tree_manifest(resolved)
    if stat.S_IMODE(resolved.lstat().st_mode) != 0o555:
        raise RuntimeError("modules cache root is not frozen 0555")
    _require_frozen_manifest(manifest)
    return manifest


def _publish_frozen_tree(partial: Path, destination: Path) -> dict[str, Any]:
    partial_root = _require_real_directory(partial, "partial modules cache")
    if ".partial" not in partial_root.name:
        raise RuntimeError(f"publication source is not an explicit .partial: {partial_root}")
    destination = _future_path(destination, "published modules cache")
    before = _tree_manifest(partial_root)
    _require_frozen_manifest(before)
    if stat.S_IMODE(partial_root.lstat().st_mode) != 0o555:
        raise RuntimeError("partial modules cache root is not frozen 0555")
    if partial_root.parent != destination.parent:
        raise RuntimeError("partial and published cache must have the same parent")
    before_stat = partial_root.lstat()
    if before_stat.st_dev != destination.parent.stat().st_dev:
        raise RuntimeError("partial and published cache are not on the same filesystem")

    os.mkdir(destination, 0o700)
    placeholder = _identity(destination.lstat())
    committed = False
    try:
        if any(destination.iterdir()) or _identity(destination.lstat()) != placeholder:
            raise RuntimeError("publication placeholder changed before rename")
        os.rename(partial_root, destination)
        committed = True
        after_stat = destination.lstat()
        if (after_stat.st_dev, after_stat.st_ino) != (
            before_stat.st_dev,
            before_stat.st_ino,
        ):
            raise RuntimeError("published tree is not the same root inode as the partial")
        after = _tree_manifest(destination)
        _require_frozen_manifest(after)
        if after["entries"] != before["entries"]:
            raise RuntimeError("published modules cache bytes or modes changed during rename")
        directory = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if not committed:
            try:
                destination.rmdir()
            except OSError:
                pass
        raise

    body = {
        "content_manifest_sha256": after["content_manifest_sha256"],
        "destination": str(destination),
        "destination_root_identity": after["root_identity"],
        "format": PUBLICATION_FORMAT,
        "no_replace_placeholder_identity": list(placeholder),
        "partial": str(partial_root),
        "partial_root_identity": before["root_identity"],
        "same_root_inode": True,
        "tree_entry_count": len(after["entries"]),
    }
    body["manifest_sha256"] = _sha256(_canonical_json(body))
    return body


def _copy_tree_mutable(source: Path, destination: Path) -> None:
    source_root = _require_real_directory(source, "negative source cache")
    destination = _future_path(destination, "negative partial cache")
    destination.mkdir(mode=0o700)
    for directory, names, leaves in os.walk(source_root, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(source_root)
        target_directory = destination / relative_directory
        for name in sorted(names):
            child = directory_path / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"negative source has non-directory child: {child}")
            (target_directory / name).mkdir(mode=0o700)
        for name in sorted(leaves):
            child = directory_path / name
            payload, _, _, _ = _stable_file(
                child, "negative-copy source", maximum_bytes=MAX_CAPTURE_BYTES
            )
            target = target_directory / name
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())


def _loaded_module_provenance(modules_cache_root: Path) -> list[dict[str, Any]]:
    root = modules_cache_root.resolve(strict=True)
    records: list[dict[str, Any]] = []
    for name, module in sorted(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            continue
        try:
            resolved = Path(filename).resolve(strict=True)
            relative = resolved.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        _, digest, size, _ = _stable_file(
            resolved, "loaded dynamic module", maximum_bytes=MAX_CAPTURE_BYTES
        )
        records.append(
            {
                "file": str(resolved),
                "module_name": name,
                "relative": relative,
                "sha256": digest,
                "size": size,
            }
        )
    return records


def _ref_evidence(hub_cache_root: Path, repository: str, revision: str) -> dict[str, Any]:
    repo_root = hub_cache_root / ("models--" + repository.replace("/", "--"))
    ref = repo_root / "refs" / "main"
    payload, digest, size, identity = _stable_file(
        ref, "remote-code refs/main", maximum_bytes=4096
    )
    if payload not in {revision.encode(), (revision + "\n").encode()}:
        raise RuntimeError(
            f"remote-code refs/main does not select expected revision: {payload!r}"
        )
    if _identity(ref.lstat()) != identity:
        raise RuntimeError("remote-code refs/main changed after read")
    return {
        "path": str(ref.resolve(strict=True)),
        "raw_sha256": digest,
        "revision": revision,
        "size": size,
        "value": revision,
    }


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_authority(verifier, source_evidence: Path, repository: str):
    value = verifier._load_json_file(source_evidence, "Hub snapshot verification")
    rows = [
        row
        for row in value.get("remote_code", [])
        if isinstance(row, dict) and row.get("repository") == repository
    ]
    if len(rows) != 1:
        raise RuntimeError(f"source evidence has no unique remote row for {repository}")
    row = rows[0]
    closure = row.get("module_closure")
    entries = row.get("entry_modules")
    if not isinstance(closure, list) or not isinstance(entries, list):
        raise RuntimeError("source evidence remote row lacks module closure or entries")
    return row, closure, entries


def _install_import_audit() -> tuple[list[dict[str, Any]], dict[str, bool]]:
    started = time.monotonic_ns()
    events: list[dict[str, Any]] = []
    state = {"truncated": False}

    def hook(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "import":
            return
        if len(events) >= MAX_IMPORT_EVENTS:
            state["truncated"] = True
            return
        name = arguments[0] if arguments and isinstance(arguments[0], str) else "<unknown>"
        filename = (
            arguments[1]
            if len(arguments) > 1 and isinstance(arguments[1], str)
            else None
        )
        events.append(
            {
                "file": filename,
                "module": name,
                "offset_ns": time.monotonic_ns() - started,
            }
        )

    sys.addaudithook(hook)
    return events, state


def _materialize_remote(
    repository: str, revision: str, hub_cache_root: Path
) -> dict[str, str]:
    import transformers
    from transformers import AutoConfig

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"wrong Transformers version: {transformers.__version__}; "
            f"expected={TRANSFORMERS_VERSION}"
        )
    config = AutoConfig.from_pretrained(
        repository,
        cache_dir=str(hub_cache_root),
        local_files_only=True,
        revision=revision,
        trust_remote_code=True,
    )
    return {
        "class": config.__class__.__name__,
        "module": config.__class__.__module__,
        "transformers_version": transformers.__version__,
    }


def _materialize_processor(model_path: Path) -> dict[str, str]:
    import transformers
    from transformers import AutoProcessor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"wrong Transformers version: {transformers.__version__}; "
            f"expected={TRANSFORMERS_VERSION}"
        )
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True
    )
    return {
        "class": processor.__class__.__name__,
        "module": processor.__class__.__module__,
        "transformers_version": transformers.__version__,
    }


def _runtime_preconditions(modules_cache: Path) -> None:
    expected = _require_real_directory(modules_cache, "HF modules cache")
    configured = os.environ.get("HF_MODULES_CACHE")
    if not configured or Path(configured).resolve(strict=True) != expected:
        raise RuntimeError("HF_MODULES_CACHE does not select the lane cache")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("CPU lane requires exact HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU lane must set CUDA_VISIBLE_DEVICES to the empty string")
    if not sys.dont_write_bytecode or not sys.flags.safe_path:
        raise RuntimeError("CPU lane interpreter must use effective -B and -P")
    if not sys.path or Path(sys.path[0]).resolve(strict=True) != expected:
        raise RuntimeError("HF_MODULES_CACHE must be the exact first sys.path entry")


def _result_base(format_name: str, lane: str) -> dict[str, Any]:
    return {
        "argv": list(sys.argv),
        "environment": _environment_authority(dict(os.environ)),
        "format": format_name,
        "lane": lane,
        "status": "running",
        "traceback": "",
    }


def _run_child(
    command: list[str], environment: dict[str, str], cwd: Path, timeout: int
) -> dict[str, Any]:
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        return {
            "argv": command,
            "duration_ns": time.monotonic_ns() - started,
            "returncode": completed.returncode,
            "stderr_sha256": _sha256(stderr),
            "stderr_size": len(stderr),
            "stdout_sha256": _sha256(stdout),
            "stdout_size": len(stdout),
            "timeout": False,
        }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, bytes) else (error.stdout or "").encode()
        stderr = error.stderr if isinstance(error.stderr, bytes) else (error.stderr or "").encode()
        return {
            "argv": command,
            "duration_ns": time.monotonic_ns() - started,
            "returncode": None,
            "stderr_sha256": _sha256(stderr),
            "stderr_size": len(stderr),
            "stdout_sha256": _sha256(stdout),
            "stdout_size": len(stdout),
            "timeout": True,
        }


def _lane_environment(
    lane_root: Path, hub_cache_root: Path, modules_cache: Path
) -> dict[str, str]:
    environment: dict[str, str] = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": str(lane_root / "hf-home"),
        "HF_HUB_CACHE": str(hub_cache_root),
        "HF_HUB_OFFLINE": "1",
        "HF_MODULES_CACHE": str(modules_cache),
        "HOME": str(lane_root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "PATH": os.environ.get(
            "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(modules_cache),
        "PYTHONSAFEPATH": "1",
        "TMPDIR": str(lane_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(lane_root / "xdg-cache"),
    }
    for name in ("LD_LIBRARY_PATH", "SSL_CERT_FILE"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _consumer_command(
    runner: Path,
    args: argparse.Namespace,
    modules_cache: Path,
    expected: str,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-P",
        str(runner),
        "consume",
        "--lane",
        args.lane,
        "--kind",
        "processor" if args.lane == "processor-positive" else "remote",
        "--modules-cache-root",
        str(modules_cache),
        "--hub-cache-root",
        str(args.hub_cache_root),
        "--processor-model-path",
        str(args.processor_model_path),
        "--snapshot-verification",
        str(args.snapshot_verification),
        "--cas-preflight",
        str(args.cas_preflight),
        "--modules-verifier",
        str(args.modules_verifier),
        "--repository",
        args.repository,
        "--revision",
        args.revision,
        "--expected",
        expected,
        "--result",
        str(result_path),
    ]


def _run_consumer(args: argparse.Namespace) -> int:
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    modules_cache = _require_real_directory(args.modules_cache_root, "published modules cache")
    hub_cache = _require_real_directory(args.hub_cache_root, "Hub cache")
    result = _result_base(CONSUMER_FORMAT, args.lane)
    events: list[dict[str, Any]] = []
    event_state = {"truncated": False}
    exit_code = 1
    try:
        _runtime_preconditions(modules_cache)
        before = _tree_manifest(modules_cache)
        _require_frozen_manifest(before)
        if stat.S_IMODE(modules_cache.lstat().st_mode) != 0o555:
            raise RuntimeError("published modules cache root is not frozen 0555")
        mountinfo = _mountinfo_evidence()
        result["mountinfo"] = mountinfo
        result["modules_cache_authority"] = _read_only_authority(
            modules_cache, mountinfo, "published HF modules cache"
        )
        events, event_state = _install_import_audit()

        if args.kind == "processor":
            if args.expected != "pass":
                raise RuntimeError("processor consumer supports only expected=pass")
            result["load"] = _materialize_processor(args.processor_model_path)
            provenance = _loaded_module_provenance(modules_cache)
            if not provenance:
                raise RuntimeError("AutoProcessor loaded no module from HF_MODULES_CACHE")
            result["loaded_module_provenance"] = provenance
            result["verification_scope"] = (
                "processor-generated-code readonly reuse; remote C-RADIO CAS "
                "binding is the remote lane"
            )
        else:
            verifier = _load_module(
                args.modules_verifier.resolve(strict=True),
                "hf_modules_cache_verifier_consumer",
            )
            remote_row, closure, entries = _remote_authority(
                verifier, args.snapshot_verification, args.repository
            )
            if remote_row.get("revision") != args.revision:
                raise RuntimeError("source evidence revision differs from consumer revision")
            result["load"] = _materialize_remote(
                args.repository, args.revision, hub_cache
            )
            observation = verifier.capture_execution_observation(
                modules_cache,
                args.repository,
                args.revision,
                closure,
                entries,
            )
            verifier.write_canonical_noreplace(args.observation, observation)
            try:
                verification = verifier.verify(
                    modules_cache,
                    hub_cache,
                    args.snapshot_verification,
                    args.observation,
                    args.repository,
                    args.revision,
                    TRANSFORMERS_VERSION,
                    args.cas_preflight,
                )
            except RuntimeError as error:
                expected_text = {
                    "byte-tamper-reject": "do not match verified Hub CAS",
                    "stale-replay-reject": "directory set mismatch",
                }.get(args.expected)
                if expected_text is None or expected_text not in str(error):
                    raise
                result["negative_rejection"] = {
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
            else:
                if args.expected != "pass":
                    raise RuntimeError(
                        f"negative cache was incorrectly accepted: expected={args.expected}"
                    )
                result["verification"] = verification
            result["loaded_module_provenance"] = _loaded_module_provenance(
                modules_cache
            )

        after = _tree_manifest(modules_cache)
        if after["entries"] != before["entries"]:
            raise RuntimeError(
                "published modules cache changed while the readonly consumer loaded it"
            )
        result["cache_before"] = before
        result["cache_after"] = after
        result["expected"] = args.expected
        result["status"] = "pass"
        exit_code = 0
    except BaseException:
        result["status"] = "fail"
        result["traceback"] = traceback.format_exc()
        try:
            result["cache_after_failure"] = _tree_manifest(modules_cache)
        except BaseException:
            result["cache_manifest_traceback"] = traceback.format_exc()
    finally:
        result["import_trace"] = {
            "events": events,
            "truncated": event_state["truncated"],
        }
        result["duration_ns"] = time.monotonic_ns() - started_mono
        result["finished_wall_time_ns"] = time.time_ns()
        result["started_wall_time_ns"] = started_wall
        result["manifest_sha256"] = _sha256(_canonical_json(result))
        _write_json_noreplace(args.result, result)
    return exit_code


def _run_internal_lane(args: argparse.Namespace) -> int:
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    runner = Path(__file__).resolve(strict=True)
    lane_root = _require_real_directory(args.lane_root, "lane root")
    partial = lane_root / ".hf-modules.partial"
    published = lane_root / "hf-modules"
    partial.mkdir(mode=0o700)
    result = _result_base(LANE_FORMAT, args.lane)
    events: list[dict[str, Any]] = []
    event_state = {"truncated": False}
    exit_code = 1
    try:
        _runtime_preconditions(partial)
        hub_cache = _require_real_directory(args.hub_cache_root, "Hub cache root")
        processor_model = _require_real_directory(
            args.processor_model_path, "processor model path"
        )
        verifier_path = args.modules_verifier.resolve(strict=True)
        mountinfo = _mountinfo_evidence()
        result["mountinfo"] = mountinfo
        result["shared_authorities"] = [
            _read_only_authority(hub_cache, mountinfo, "Hub cache"),
            _read_only_authority(processor_model, mountinfo, "processor model"),
            _read_only_authority(verifier_path, mountinfo, "modules verifier"),
            _read_only_authority(
                args.snapshot_verification.resolve(strict=True),
                mountinfo,
                "snapshot verification",
            ),
            _read_only_authority(
                args.cas_preflight.resolve(strict=True), mountinfo, "CAS preflight"
            ),
            _read_only_authority(runner, mountinfo, "CPU lane runner"),
        ]
        result["refs"] = _ref_evidence(hub_cache, args.repository, args.revision)
        result["cache_before"] = _tree_manifest(partial)
        events, event_state = _install_import_audit()

        if args.lane == "processor-positive":
            result["materialization"] = _materialize_processor(processor_model)
            provenance = _loaded_module_provenance(partial)
            if not provenance:
                raise RuntimeError("AutoProcessor loaded no module from HF_MODULES_CACHE")
            result["loaded_module_provenance"] = provenance
            frozen = _freeze_tree(partial)
            result["prepublication_attestation"] = {
                "cache_manifest": frozen,
                "scope": "processor-generated-code bytes and readonly publication",
            }
        else:
            verifier = _load_module(verifier_path, "hf_modules_cache_verifier_materializer")
            remote_row, closure, entries = _remote_authority(
                verifier, args.snapshot_verification, args.repository
            )
            if remote_row.get("revision") != args.revision:
                raise RuntimeError("source evidence revision differs from lane revision")
            result["materialization"] = _materialize_remote(
                args.repository, args.revision, hub_cache
            )
            observation_path = lane_root / "materializer-observation.json"
            observation = verifier.capture_execution_observation(
                partial, args.repository, args.revision, closure, entries
            )
            verifier.write_canonical_noreplace(observation_path, observation)
            frozen = _freeze_tree(partial)
            verification = verifier.verify(
                partial,
                hub_cache,
                args.snapshot_verification,
                observation_path,
                args.repository,
                args.revision,
                TRANSFORMERS_VERSION,
                args.cas_preflight,
            )
            result["prepublication_attestation"] = {
                "cache_manifest": frozen,
                "verification": verification,
            }
            result["loaded_module_provenance"] = _loaded_module_provenance(partial)

        result["publication"] = _publish_frozen_tree(partial, published)

        baseline_result = lane_root / "readonly-consumer.json"
        baseline_observation = lane_root / "readonly-consumer-observation.json"
        baseline_command = _consumer_command(
            runner, args, published, "pass", baseline_result
        ) + ["--observation", str(baseline_observation)]
        baseline_environment = _lane_environment(lane_root, hub_cache, published)
        baseline_invocation = _run_child(
            baseline_command,
            baseline_environment,
            lane_root,
            args.timeout_seconds,
        )
        baseline = _load_json(baseline_result, "readonly consumer result")
        if baseline_invocation["returncode"] != 0 or baseline.get("status") != "pass":
            raise RuntimeError("fresh readonly consumer failed; see consumer evidence")
        _require_consumer_publication_join(result["publication"], baseline)
        result["readonly_consumer"] = _evidence_reference(
            baseline_result, baseline, baseline_invocation
        )

        if args.lane in {"byte-tamper", "stale-replay"}:
            negative_partial = lane_root / ".hf-modules-negative.partial"
            negative_published = lane_root / "hf-modules-negative"
            _copy_tree_mutable(published, negative_partial)
            verifier = _load_module(verifier_path, "hf_modules_cache_verifier_negative")
            remote_row, closure, _ = _remote_authority(
                verifier, args.snapshot_verification, args.repository
            )
            revision_root = (
                negative_partial
                / verifier._generated_repository_relative(args.repository)
                / args.revision
            )
            if args.lane == "byte-tamper":
                source = sorted(closure)[-1]
                target = revision_root / PurePosixPath(source)
                payload, _, _, _ = _stable_file(target, "byte-tamper target")
                target.write_bytes(payload + b"\n# intentional qualification tamper\n")
                expected = "byte-tamper-reject"
                mutation = {"kind": "byte-append", "target": str(target)}
            else:
                stale_revision = "a" * 40
                if stale_revision == args.revision:
                    stale_revision = "c" * 40
                stale = revision_root.parent / stale_revision
                _copy_tree_mutable(revision_root, stale)
                expected = "stale-replay-reject"
                mutation = {"kind": "extra-revision", "revision": stale_revision}
            negative_frozen = _freeze_tree(negative_partial)
            negative_publication = _publish_frozen_tree(
                negative_partial, negative_published
            )
            negative_result = lane_root / "negative-consumer.json"
            negative_observation = lane_root / "negative-consumer-observation.json"
            negative_command = _consumer_command(
                runner, args, negative_published, expected, negative_result
            ) + ["--observation", str(negative_observation)]
            negative_environment = _lane_environment(
                lane_root, hub_cache, negative_published
            )
            negative_invocation = _run_child(
                negative_command,
                negative_environment,
                lane_root,
                args.timeout_seconds,
            )
            negative = _load_json(negative_result, "negative consumer result")
            if negative_invocation["returncode"] != 0 or negative.get("status") != "pass":
                raise RuntimeError("negative consumer did not prove the expected rejection")
            _require_consumer_publication_join(negative_publication, negative)
            result["negative_lane"] = {
                "attestation": negative_frozen,
                "consumer": _evidence_reference(
                    negative_result, negative, negative_invocation
                ),
                "mutation": mutation,
                "publication": negative_publication,
            }

        result["status"] = "pass"
        exit_code = 0
    except BaseException:
        result["status"] = "fail"
        result["traceback"] = traceback.format_exc()
        for name, path in (
            ("partial", partial),
            ("published", published),
        ):
            try:
                if path.is_dir() and not path.is_symlink():
                    result[f"{name}_after_failure"] = _tree_manifest(path)
            except BaseException:
                result[f"{name}_manifest_traceback"] = traceback.format_exc()
    finally:
        result["import_trace"] = {
            "events": events,
            "truncated": event_state["truncated"],
        }
        result["duration_ns"] = time.monotonic_ns() - started_mono
        result["finished_wall_time_ns"] = time.time_ns()
        result["started_wall_time_ns"] = started_wall
        result["manifest_sha256"] = _sha256(_canonical_json(result))
        _write_json_noreplace(args.result, result)
    return exit_code


def _diagnostic_matrix(
    output_root: Path,
    hub_cache_root: Path,
    processor_model_path: Path,
    snapshot_verification: Path,
    modules_verifier: Path,
    runner: Path,
    cas_preflight: Path,
) -> dict[str, Any]:
    if output_root.exists() and not output_root.is_symlink():
        canonical_output_root = output_root.resolve(strict=True)
    else:
        canonical_output_root = output_root.parent.resolve(strict=True) / output_root.name
    shared_reads = [
        hub_cache_root.resolve(strict=True),
        processor_model_path.resolve(strict=True),
        snapshot_verification.resolve(strict=True),
        modules_verifier.resolve(strict=True),
        runner.resolve(strict=True),
        cas_preflight.resolve(strict=True),
    ]
    lane_rows: list[dict[str, Any]] = []
    writable_roots: list[Path] = []
    for lane in LANES:
        lane_root = canonical_output_root / "lanes" / lane
        writable_roots.append(lane_root)
        lane_rows.append(
            {
                "cpu_slots": 1,
                "gpu_slots": 0,
                "hf_modules_cache_isolated": True,
                "lane": lane,
                "writable_root": str(lane_root),
            }
        )
    for index, left in enumerate(writable_roots):
        for right in writable_roots[index + 1 :]:
            if _inside(left, right) or _inside(right, left):
                raise RuntimeError(f"lane writable roots overlap: {left}; {right}")
        for shared in shared_reads:
            if _inside(left, shared) or _inside(shared, left):
                raise RuntimeError(
                    f"lane writable root overlaps shared authority: {left}; {shared}"
                )
    return {
        "cas_full_hash_count": 1,
        "hazards": [],
        "lanes": lane_rows,
        "negative_mutation_scope": "attempt copy only; sealed CAS is shared read-only",
        "shared_read_authorities": [str(path) for path in shared_reads],
        "shared_writable_authorities": [],
    }


def _lane_command(
    runner: Path,
    lane: str,
    lane_root: Path,
    hub_cache_root: Path,
    processor_model_path: Path,
    snapshot_verification: Path,
    cas_preflight: Path,
    modules_verifier: Path,
    repository: str,
    revision: str,
    timeout_seconds: int,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-P",
        str(runner),
        "lane",
        "--lane",
        lane,
        "--lane-root",
        str(lane_root),
        "--hub-cache-root",
        str(hub_cache_root),
        "--processor-model-path",
        str(processor_model_path),
        "--snapshot-verification",
        str(snapshot_verification),
        "--cas-preflight",
        str(cas_preflight),
        "--modules-verifier",
        str(modules_verifier),
        "--repository",
        repository,
        "--revision",
        revision,
        "--timeout-seconds",
        str(timeout_seconds),
        "--result",
        str(lane_root / "result.json"),
    ]


def _run_bundle(args: argparse.Namespace) -> int:
    if args.max_workers < 1 or args.max_workers > len(LANES):
        raise RuntimeError(f"max-workers must be in [1, {len(LANES)}]")
    if not HEX40.fullmatch(args.revision):
        raise RuntimeError("revision must be a 40-character lowercase commit")
    runner = Path(__file__).resolve(strict=True)
    hub_cache = _require_real_directory(args.hub_cache_root, "Hub cache root")
    processor_model = _require_real_directory(
        args.processor_model_path, "processor model path"
    )
    snapshot_verification = args.snapshot_verification.resolve(strict=True)
    modules_verifier = args.modules_verifier.resolve(strict=True)
    output_root = _future_path(args.output_root, "bundle output root")
    output_root.mkdir(mode=0o700)
    (output_root / "lanes").mkdir()
    cas_preflight_path = output_root / "CAS_PREFLIGHT.json"

    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    verifier = _load_module(modules_verifier, "hf_modules_cache_verifier_preflight")
    cas_preflight = verifier.build_cas_preflight(
        hub_cache, snapshot_verification, args.repository, args.revision
    )
    verifier.write_canonical_noreplace(cas_preflight_path, cas_preflight)
    matrix = _diagnostic_matrix(
        output_root,
        hub_cache,
        processor_model,
        snapshot_verification,
        modules_verifier,
        runner,
        cas_preflight_path,
    )
    mountinfo = _mountinfo_evidence()
    shared_authorities = [
        _read_only_authority(hub_cache, mountinfo, "Hub cache"),
        _read_only_authority(processor_model, mountinfo, "processor model"),
        _read_only_authority(snapshot_verification, mountinfo, "snapshot verification"),
        _read_only_authority(modules_verifier, mountinfo, "modules verifier"),
        _read_only_authority(runner, mountinfo, "CPU lane runner"),
        _read_only_authority(cas_preflight_path, mountinfo, "CAS preflight"),
    ]

    invocations: dict[str, dict[str, Any]] = {}
    futures: dict[concurrent.futures.Future, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for lane in LANES:
            lane_root = output_root / "lanes" / lane
            lane_root.mkdir()
            for name in ("hf-home", "home", "tmp", "xdg-cache"):
                (lane_root / name).mkdir()
            partial = lane_root / ".hf-modules.partial"
            environment = _lane_environment(lane_root, hub_cache, partial)
            command = _lane_command(
                runner,
                lane,
                lane_root,
                hub_cache,
                processor_model,
                snapshot_verification,
                cas_preflight_path,
                modules_verifier,
                args.repository,
                args.revision,
                args.timeout_seconds,
            )
            futures[
                pool.submit(
                    _run_child,
                    command,
                    environment,
                    lane_root,
                    args.timeout_seconds * 3,
                )
            ] = lane
        for future in concurrent.futures.as_completed(futures):
            invocations[futures[future]] = future.result()

    lane_results: list[dict[str, Any]] = []
    all_passed = True
    for lane in LANES:
        invocation = invocations[lane]
        result_path = output_root / "lanes" / lane / "result.json"
        record: dict[str, Any] = {"invocation": invocation, "lane": lane}
        decoded: dict[str, Any] | None = None
        try:
            decoded = _load_json(result_path, f"{lane} result")
            record["result"] = _evidence_reference(result_path, decoded)
        except BaseException:
            record["result_traceback"] = traceback.format_exc()
        passed = (
            invocation["returncode"] == 0
            and not invocation["timeout"]
            and decoded is not None
            and decoded.get("status") == "pass"
        )
        record["passed"] = passed
        all_passed = all_passed and passed
        lane_results.append(record)

    result = {
        "argv": list(sys.argv),
        "cas_preflight_manifest_sha256": cas_preflight["manifest_sha256"],
        "cas_preflight_path": str(cas_preflight_path),
        "diagnostic_matrix": matrix,
        "duration_ns": time.monotonic_ns() - started_mono,
        "environment": _environment_authority(dict(os.environ)),
        "finished_wall_time_ns": time.time_ns(),
        "format": BUNDLE_FORMAT,
        "hub_cache_root": str(hub_cache),
        "lanes": lane_results,
        "max_workers": args.max_workers,
        "mountinfo": mountinfo,
        "output_root": str(output_root),
        "processor_model_path": str(processor_model),
        "repository": args.repository,
        "revision": args.revision,
        "shared_authorities": shared_authorities,
        "started_wall_time_ns": started_wall,
        "status": "pass" if all_passed else "fail",
        "timeout_seconds": args.timeout_seconds,
    }
    result["manifest_sha256"] = _sha256(_canonical_json(result))
    _write_json_noreplace(output_root / "bundle-result.json", result)
    print(
        "HF_DYNAMIC_CODE_CPU_BUNDLE|"
        f"status={result['status']}|lanes={len(LANES)}|"
        f"manifest={result['manifest_sha256']}|output={output_root}"
    )
    return 0 if all_passed else 1


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hub-cache-root", type=Path, required=True)
    parser.add_argument("--processor-model-path", type=Path, required=True)
    parser.add_argument("--snapshot-verification", type=Path, required=True)
    parser.add_argument("--cas-preflight", type=Path, required=True)
    parser.add_argument("--modules-verifier", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("--output-root", type=Path, required=True)
    bundle.add_argument("--hub-cache-root", type=Path, required=True)
    bundle.add_argument("--processor-model-path", type=Path, required=True)
    bundle.add_argument("--snapshot-verification", type=Path, required=True)
    bundle.add_argument("--modules-verifier", type=Path, required=True)
    bundle.add_argument("--repository", required=True)
    bundle.add_argument("--revision", required=True)
    bundle.add_argument("--max-workers", type=int, default=len(LANES))
    bundle.add_argument("--timeout-seconds", type=int, default=900)

    lane = subparsers.add_parser("lane")
    lane.add_argument("--lane", choices=LANES, required=True)
    lane.add_argument("--lane-root", type=Path, required=True)
    _add_shared_arguments(lane)
    lane.add_argument("--timeout-seconds", type=int, required=True)
    lane.add_argument("--result", type=Path, required=True)

    consume = subparsers.add_parser("consume")
    consume.add_argument("--lane", choices=LANES, required=True)
    consume.add_argument("--kind", choices=("remote", "processor"), required=True)
    consume.add_argument("--modules-cache-root", type=Path, required=True)
    _add_shared_arguments(consume)
    consume.add_argument(
        "--expected",
        choices=("pass", "byte-tamper-reject", "stale-replay-reject"),
        required=True,
    )
    consume.add_argument("--observation", type=Path, required=True)
    consume.add_argument("--result", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "bundle":
        code = _run_bundle(args)
    elif args.command == "lane":
        code = _run_internal_lane(args)
    else:
        code = _run_consumer(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
