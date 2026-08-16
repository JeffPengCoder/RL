#!/usr/bin/env python3
"""Verify a complete Hugging Face cache snapshot and dynamic-code closure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


FORMAT = "hf-snapshot-root-mount-verification-v3"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REPO_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
MAX_CAPTURE_BYTES = 32 * 1024 * 1024


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


def _read_stable_blob(path: Path) -> tuple[str, str, int, bytes | None, tuple[int, ...]]:
    name = path.name
    if HEX64.fullmatch(name):
        algorithm = "sha256"
        digest = hashlib.sha256()
    elif HEX40.fullmatch(name):
        algorithm = "git-blob-sha1"
        digest = hashlib.sha1(usedforsecurity=False)
    else:
        raise RuntimeError(f"unsupported Hugging Face blob identity: {name}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"blob is not a regular file: {path}")
        if algorithm == "git-blob-sha1":
            digest.update(f"blob {before.st_size}\0".encode())
        capture = bytearray() if before.st_size <= MAX_CAPTURE_BYTES else None
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
            if capture is not None:
                capture.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"blob changed while being read: {path}")
    finally:
        os.close(descriptor)

    actual = digest.hexdigest()
    if actual != name:
        raise RuntimeError(
            f"blob digest mismatch: {path}; expected={name}; actual={actual}"
        )
    return algorithm, actual, before.st_size, bytes(capture) if capture is not None else None, _identity(before)


def _read_stable_small_file(
    path: Path, label: str, *, maximum_bytes: int = 4096
) -> tuple[bytes, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} is missing, inaccessible, or a symlink: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")
        if before.st_size > maximum_bytes:
            raise RuntimeError(
                f"{label} is unexpectedly large: {path}; bytes={before.st_size}"
            )
        payload = bytearray()
        while chunk := os.read(descriptor, 4096):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    return bytes(payload), _identity(before)


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


def _snapshot_entries(snapshot: Path) -> list[Path]:
    entries: list[Path] = []
    for directory, names, files in os.walk(snapshot, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            child = directory_path / name
            metadata = child.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise RuntimeError(f"snapshot directory is not a real directory: {child}")
        for name in files:
            entries.append(directory_path / name)
    return sorted(entries, key=lambda item: item.relative_to(snapshot).as_posix())


def _verify_snapshot(repo_root: Path, revision: str) -> dict[str, Any]:
    root = _require_real_directory(repo_root, "repository root")
    if not HEX40.fullmatch(revision):
        raise RuntimeError(f"revision must be a 40-character commit: {revision}")
    blobs = _require_real_directory(root / "blobs", "blob root")
    snapshot = _require_real_directory(root / "snapshots" / revision, "snapshot")

    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    observed: list[tuple[Path, str, Path, tuple[int, ...]]] = []
    cached: dict[Path, tuple[str, str, int, bytes | None, tuple[int, ...]]] = {}
    for entry in _snapshot_entries(snapshot):
        metadata = entry.lstat()
        if not stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"snapshot leaf is not a symlink: {entry}")
        link_value = os.readlink(entry)
        if os.path.isabs(link_value):
            raise RuntimeError(f"snapshot link is absolute: {entry} -> {link_value}")
        target = (entry.parent / link_value).resolve(strict=True)
        if target.parent != blobs or not (HEX40.fullmatch(target.name) or HEX64.fullmatch(target.name)):
            raise RuntimeError(
                f"snapshot link does not resolve to the exact blobs namespace: {entry} -> {target}"
            )
        if target.is_symlink() or not stat.S_ISREG(target.lstat().st_mode):
            raise RuntimeError(f"snapshot target is not a real regular blob: {target}")
        if target not in cached:
            cached[target] = _read_stable_blob(target)
        algorithm, digest, size, payload, identity = cached[target]
        relative = entry.relative_to(snapshot).as_posix()
        if payload is not None:
            payloads[relative] = payload
        records.append(
            {
                "blob_algorithm": algorithm,
                "blob_digest": digest,
                "link": link_value,
                "path": relative,
                "size": size,
                "target_relative": target.relative_to(root).as_posix(),
            }
        )
        observed.append((entry, link_value, target, identity))

    for entry, link_value, target, identity in observed:
        if os.readlink(entry) != link_value:
            raise RuntimeError(f"snapshot link changed during verification: {entry}")
        if _identity(target.lstat()) != identity:
            raise RuntimeError(f"blob identity changed after verification: {target}")
    final_entries = [
        entry.relative_to(snapshot).as_posix() for entry in _snapshot_entries(snapshot)
    ]
    if final_entries != [record["path"] for record in records]:
        raise RuntimeError("snapshot entry set changed during verification")

    content = {"files": records, "revision": revision}
    return {
        "blob_count": len(cached),
        "content_manifest_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
        "files": records,
        "payloads": payloads,
        "repo_root": str(root),
        "revision": revision,
        "snapshot": snapshot,
        "total_referenced_bytes": sum(record["size"] for record in records),
    }


def _load_json(payloads: dict[str, bytes], name: str) -> Any:
    payload = payloads.get(name)
    if payload is None:
        raise RuntimeError(f"verified JSON payload is unavailable: {name}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid JSON in verified snapshot file: {name}: {error}") from error


def _walk_auto_maps(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    found: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            if key == "auto_map":
                if not isinstance(child, dict) or not child:
                    raise RuntimeError(f"auto_map is not a non-empty object at {'.'.join(child_path)}")
                found.append((child_path, child))
            found.extend(_walk_auto_maps(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_auto_maps(child, path + (str(index),)))
    return found


def _module_path(reference: str) -> tuple[str | None, str]:
    if "--" in reference:
        repo_id, class_reference = reference.rsplit("--", 1)
        if not REPO_ID.fullmatch(repo_id):
            raise RuntimeError(f"invalid remote auto_map repository: {reference!r}")
    else:
        repo_id, class_reference = None, reference
    if "." not in class_reference:
        raise RuntimeError(f"invalid auto_map class reference: {reference!r}")
    module, class_name = class_reference.rsplit(".", 1)
    parts = module.split(".")
    if not class_name.isidentifier() or not parts or any(not part.isidentifier() for part in parts):
        raise RuntimeError(f"invalid auto_map Python reference: {reference!r}")
    return repo_id, "/".join(parts) + ".py"


def _collect_auto_map_references(
    payloads: dict[str, bytes], config_names: tuple[str, ...]
) -> tuple[list[dict[str, str]], set[str], dict[str, set[str]]]:
    records: list[dict[str, str]] = []
    local_modules: set[str] = set()
    remote_modules: dict[str, set[str]] = {}
    for config_name in config_names:
        value = _load_json(payloads, config_name)
        maps = _walk_auto_maps(value)
        if not maps:
            raise RuntimeError(f"approved config has no auto_map: {config_name}")
        for json_path, mapping in maps:
            for auto_class, raw in sorted(mapping.items()):
                values = raw if isinstance(raw, list) else [raw]
                for reference in values:
                    if reference is None:
                        continue
                    if not isinstance(reference, str):
                        raise RuntimeError(
                            f"invalid auto_map value at {config_name}:{'.'.join(json_path)}:{auto_class}"
                        )
                    repo_id, module = _module_path(reference)
                    kind = "remote" if repo_id else "local"
                    records.append(
                        {
                            "auto_class": str(auto_class),
                            "config": config_name,
                            "json_path": "$." + ".".join(json_path),
                            "kind": kind,
                            "module": module,
                            "reference": reference,
                            "repository": repo_id or "",
                        }
                    )
                    if repo_id:
                        remote_modules.setdefault(repo_id, set()).add(module)
                    else:
                        local_modules.add(module)
    return records, local_modules, remote_modules


def _relative_imports(module_path: str, payload: bytes, available: set[str]) -> set[str]:
    try:
        tree = ast.parse(payload, filename=module_path)
    except (SyntaxError, ValueError) as error:
        raise RuntimeError(f"cannot parse verified dynamic module {module_path}: {error}") from error
    package = list(PurePosixPath(module_path).parent.parts)
    if package == ["."]:
        package = []
    imports: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        if node.level - 1 > len(package):
            raise RuntimeError(f"relative import escapes dynamic-module root: {module_path}:{node.lineno}")
        base = package[: len(package) - (node.level - 1)]
        if node.module:
            candidate_parts = base + node.module.split(".")
            candidates = ["/".join(candidate_parts) + ".py", "/".join(candidate_parts + ["__init__"]) + ".py"]
            match = next((candidate for candidate in candidates if candidate in available), None)
            if match is None:
                raise RuntimeError(
                    f"relative dynamic-module import is absent: {module_path}:{node.lineno}:{node.module}"
                )
            imports.add(match)
        else:
            for alias in node.names:
                candidate_parts = base + alias.name.split(".")
                candidates = [
                    "/".join(candidate_parts) + ".py",
                    "/".join(candidate_parts + ["__init__"]) + ".py",
                ]
                match = next(
                    (candidate for candidate in candidates if candidate in available),
                    None,
                )
                if match is not None:
                    imports.add(match)
                    continue
                package_init = "/".join(base + ["__init__"]) + ".py"
                if package_init in available:
                    imports.add(package_init)
                    continue
                raise RuntimeError(
                    "relative dynamic-module import is absent: "
                    f"{module_path}:{node.lineno}:{alias.name}"
                )
    return imports


def _dynamic_module_closure(start: set[str], payloads: dict[str, bytes]) -> list[str]:
    available = set(payloads)
    missing = sorted(start - available)
    if missing:
        raise RuntimeError(f"dynamic auto_map modules are absent: {missing}")
    closure: set[str] = set()
    queue = sorted(start)
    while queue:
        module = queue.pop(0)
        if module in closure:
            continue
        payload = payloads.get(module)
        if payload is None:
            raise RuntimeError(f"dynamic module payload was too large or unavailable: {module}")
        closure.add(module)
        for dependency in sorted(_relative_imports(module, payload, available)):
            if dependency not in closure:
                queue.append(dependency)
    return sorted(closure)


def _parse_remote_revisions(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        repo_id, separator, revision = value.partition("=")
        if not separator or not REPO_ID.fullmatch(repo_id) or not HEX40.fullmatch(revision):
            raise RuntimeError(f"invalid --remote-code-revision: {value!r}")
        if repo_id in result:
            raise RuntimeError(f"duplicate remote-code repository: {repo_id}")
        result[repo_id] = revision
    return result


def _parse_remote_manifests(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        repo_id, separator, manifest = value.partition("=")
        if not separator or not REPO_ID.fullmatch(repo_id) or not HEX64.fullmatch(manifest):
            raise RuntimeError(f"invalid --expected-remote-code-manifest: {value!r}")
        if repo_id in result:
            raise RuntimeError(f"duplicate remote-code manifest repository: {repo_id}")
        result[repo_id] = manifest
    return result


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _verify_remote_code(
    cache_root: Path | None,
    remote_modules: dict[str, set[str]],
    revisions: dict[str, str],
    expected_manifests: dict[str, str],
) -> list[dict[str, Any]]:
    if set(remote_modules) != set(revisions):
        raise RuntimeError(
            "remote auto_map repositories do not exactly match pinned revisions: "
            f"observed={sorted(remote_modules)} pinned={sorted(revisions)}"
        )
    if set(remote_modules) != set(expected_manifests):
        raise RuntimeError(
            "remote auto_map repositories do not exactly match expected manifests: "
            f"observed={sorted(remote_modules)} expected={sorted(expected_manifests)}"
        )
    if not remote_modules:
        return []
    if cache_root is None:
        raise RuntimeError("remote auto_map references require --remote-code-cache-root")
    root = _require_real_directory(cache_root, "remote-code cache root")
    results: list[dict[str, Any]] = []
    for repo_id in sorted(remote_modules):
        revision = revisions[repo_id]
        repo_root = root / _repo_cache_name(repo_id)
        refs_root = _require_real_directory(repo_root / "refs", "remote-code refs root")
        reference = refs_root / "main"
        reference_payload, reference_identity = _read_stable_small_file(
            reference, "remote-code main ref"
        )
        if reference_payload not in {
            revision.encode("ascii"),
            (revision + "\n").encode("ascii"),
        }:
            raise RuntimeError(f"remote-code main ref is not pinned to {revision}: {reference}")
        snapshot = _verify_snapshot(repo_root, revision)
        if _identity(reference.lstat()) != reference_identity:
            raise RuntimeError(f"remote-code main ref changed during verification: {reference}")
        if snapshot["content_manifest_sha256"] != expected_manifests[repo_id]:
            raise RuntimeError(
                f"remote-code snapshot manifest mismatch for {repo_id}: "
                f"expected={expected_manifests[repo_id]} "
                f"observed={snapshot['content_manifest_sha256']}"
            )
        closure = _dynamic_module_closure(remote_modules[repo_id], snapshot["payloads"])
        results.append(
            {
                "content_manifest_sha256": snapshot["content_manifest_sha256"],
                "entry_modules": sorted(remote_modules[repo_id]),
                "main_ref_relative": "refs/main",
                "main_ref_sha256": hashlib.sha256(reference_payload).hexdigest(),
                "main_ref_value": revision,
                "module_closure": closure,
                "repository": repo_id,
                "revision": revision,
                "snapshot_entry_count": len(snapshot["files"]),
            }
        )
    return results


def _verify_offline_environment(cache_root: Path | None) -> dict[str, str]:
    if cache_root is None:
        raise RuntimeError("offline dynamic-code loading requires --remote-code-cache-root")
    expected = _require_real_directory(cache_root, "remote-code cache root")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must be exactly 1 for offline dynamic-code loading")
    configured = os.environ.get("HF_HUB_CACHE")
    if not configured:
        raise RuntimeError("HF_HUB_CACHE must select the sealed remote-code cache root")
    observed = _require_real_directory(Path(configured), "HF_HUB_CACHE")
    if observed != expected:
        raise RuntimeError(
            f"HF_HUB_CACHE does not select the sealed remote-code cache root: "
            f"expected={expected} observed={observed}"
        )
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(name)
        if value and _require_real_directory(Path(value), name) != expected:
            raise RuntimeError(f"{name} conflicts with the sealed remote-code cache root")
    return {
        "hf_hub_cache": str(expected),
        "hf_hub_offline": "1",
        "transformers_offline": "1",
    }


def verify(
    model_root: Path,
    revision: str,
    expected_shards: int | None = None,
    *,
    required_paths: tuple[str, ...] = (),
    auto_map_configs: tuple[str, ...] = (),
    remote_code_cache_root: Path | None = None,
    remote_code_revisions: dict[str, str] | None = None,
    expected_remote_code_manifests: dict[str, str] | None = None,
    expected_snapshot_manifest_sha256: str | None = None,
    require_offline_env: bool = False,
) -> dict[str, Any]:
    snapshot = _verify_snapshot(model_root, revision)
    by_name = {record["path"]: record for record in snapshot["files"]}
    for required_path in required_paths:
        path = PurePosixPath(required_path)
        if not required_path or path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"invalid required snapshot path: {required_path!r}")
        if required_path not in by_name:
            raise RuntimeError(f"required snapshot path is absent: {required_path}")

    index_name = "model.safetensors.index.json"
    index = _load_json(snapshot["payloads"], index_name)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("safetensors index has no non-empty weight_map")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in weight_map.items()
    ):
        raise RuntimeError("safetensors weight_map keys and values must be non-empty strings")
    shards = sorted(set(weight_map.values()))
    if expected_shards is not None and len(shards) != expected_shards:
        raise RuntimeError(f"expected {expected_shards} safetensors shards, found {len(shards)}")
    for shard in shards:
        path = PurePosixPath(shard)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in shard
            or path.as_posix() != shard
            or not shard.endswith(".safetensors")
        ):
            raise RuntimeError(f"invalid safetensors shard name: {shard!r}")
        if shard not in by_name:
            raise RuntimeError(f"indexed safetensors shard is absent: {shard}")
    observed_safetensors = {
        name for name in by_name if name.endswith(".safetensors")
    }
    if observed_safetensors != set(shards):
        raise RuntimeError(
            "snapshot safetensors files do not exactly match the index: "
            f"indexed={shards} observed={sorted(observed_safetensors)}"
        )

    references, local_modules, remote_modules = _collect_auto_map_references(
        snapshot["payloads"], auto_map_configs
    )
    local_closure = _dynamic_module_closure(local_modules, snapshot["payloads"])
    remote_results = _verify_remote_code(
        remote_code_cache_root,
        remote_modules,
        remote_code_revisions or {},
        expected_remote_code_manifests or {},
    )
    offline_environment = (
        _verify_offline_environment(remote_code_cache_root)
        if require_offline_env
        else None
    )
    observed_manifest = snapshot["content_manifest_sha256"]
    if expected_snapshot_manifest_sha256 is not None:
        if not HEX64.fullmatch(expected_snapshot_manifest_sha256):
            raise RuntimeError("expected snapshot manifest must be 64 lowercase hex characters")
        if observed_manifest != expected_snapshot_manifest_sha256:
            raise RuntimeError(
                "snapshot manifest mismatch: "
                f"expected={expected_snapshot_manifest_sha256} observed={observed_manifest}"
            )

    result = {
        "auto_map_configs": list(auto_map_configs),
        "auto_map_references": references,
        "blob_count": snapshot["blob_count"],
        "files": snapshot["files"],
        "format": FORMAT,
        "local_dynamic_module_closure": local_closure,
        "model_root": snapshot["repo_root"],
        "offline_environment": offline_environment,
        "remote_code": remote_results,
        "required_paths": list(required_paths),
        "revision": revision,
        "safetensors_shard_count": len(shards),
        "snapshot_content_manifest_sha256": observed_manifest,
        "snapshot_entry_count": len(snapshot["files"]),
        "snapshot_relative": f"snapshots/{revision}",
        "total_referenced_bytes": snapshot["total_referenced_bytes"],
    }
    result["manifest_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _write_atomic_noreplace(path: Path, payload: bytes) -> None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--required-path", action="append", default=[])
    parser.add_argument("--auto-map-config", action="append", default=[])
    parser.add_argument("--remote-code-cache-root", type=Path)
    parser.add_argument("--remote-code-revision", action="append", default=[])
    parser.add_argument("--expected-remote-code-manifest", action="append", default=[])
    parser.add_argument("--expected-snapshot-manifest-sha256")
    parser.add_argument("--require-offline-env", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.model_root,
        args.revision,
        args.expected_shards,
        required_paths=tuple(args.required_path),
        auto_map_configs=tuple(args.auto_map_config),
        remote_code_cache_root=args.remote_code_cache_root,
        remote_code_revisions=_parse_remote_revisions(args.remote_code_revision),
        expected_remote_code_manifests=_parse_remote_manifests(
            args.expected_remote_code_manifest
        ),
        expected_snapshot_manifest_sha256=args.expected_snapshot_manifest_sha256,
        require_offline_env=args.require_offline_env,
    )
    _write_atomic_noreplace(args.output, _canonical_json(result))
    print(
        "HF_SNAPSHOT_ROOT_MOUNT_VERIFIED|"
        f"revision={result['revision']}|entries={result['snapshot_entry_count']}|"
        f"shards={result['safetensors_shard_count']}|"
        f"remote_repos={len(result['remote_code'])}|"
        f"offline={int(result['offline_environment'] is not None)}|"
        f"bytes={result['total_referenced_bytes']}|"
        f"snapshot_manifest={result['snapshot_content_manifest_sha256']}|"
        f"result_manifest={result['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
