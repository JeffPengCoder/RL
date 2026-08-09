# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create deterministic, disjoint OSWorld train/validation exact-trace manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


def _task_identity(row: dict[str, Any], *, row_number: int) -> tuple[str, str]:
    metadata = row.get("verifier_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"row {row_number} has no verifier_metadata mapping")
    osworld_task = metadata.get("osworld_task")
    if not isinstance(osworld_task, dict):
        raise ValueError(f"row {row_number} has no verifier_metadata.osworld_task")
    task_id = metadata.get("task_id") or osworld_task.get("id")
    domain = metadata.get("domain") or osworld_task.get("snapshot")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"row {row_number} has no non-empty OSWorld task ID")
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"row {row_number} has no non-empty OSWorld domain")
    if osworld_task.get("id") not in {None, task_id}:
        raise ValueError(f"row {row_number} has conflicting OSWorld task IDs")
    return task_id, domain


def load_osworld_rows(
    path: Path,
    *,
    domains: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate unique OSWorld rows, optionally filtering domains."""
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row {row_number} must be a JSON object")
            task_id, domain = _task_identity(value, row_number=row_number)
            if domains and domain not in domains:
                continue
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate OSWorld task ID {task_id!r}")
            seen_task_ids.add(task_id)
            rows.append(value)
    if not rows:
        selected = f" for domains {sorted(domains)!r}" if domains else ""
        raise ValueError(f"no OSWorld rows selected from {path}{selected}")
    return rows


def split_osworld_rows(
    rows: list[dict[str, Any]],
    *,
    validation_count: int,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a fixed validation set by stable task-ID hashing."""
    if validation_count <= 0:
        raise ValueError("validation_count must be positive")
    if validation_count >= len(rows):
        raise ValueError(
            "validation_count must leave at least one training task: "
            f"validation_count={validation_count} rows={len(rows)}"
        )

    ranked = sorted(
        enumerate(rows),
        key=lambda indexed_row: hashlib.sha256(
            f"{seed}:{_task_identity(indexed_row[1], row_number=indexed_row[0] + 1)[0]}".encode()
        ).hexdigest(),
    )
    validation_indices = {index for index, _ in ranked[:validation_count]}
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    return train, validation


def annotate_trajectory_rows(
    rows: list[dict[str, Any]],
    *,
    group_prefix: str,
    agent_name: str,
) -> list[dict[str, Any]]:
    """Add model-independent caller identity consumed by trace-aware NeMo-RL."""
    if not group_prefix:
        raise ValueError("group_prefix must not be empty")
    if not agent_name:
        raise ValueError("agent_name must not be empty")

    annotated = []
    for row_number, source in enumerate(rows, start=1):
        task_id, domain = _task_identity(source, row_number=row_number)
        row = deepcopy(source)
        legacy_fields = sorted(
            field for field in row if field.startswith("context_compaction_")
        )
        if legacy_fields:
            raise ValueError(
                f"task {task_id!r} already contains legacy identity fields: "
                + ", ".join(legacy_fields)
            )
        expected = {
            "schema_version": 1,
            "group_id": f"{group_prefix}:{domain}:{task_id}",
            "task_id": task_id,
            "rollout_index": 0,
            "attempt_index": 0,
        }
        if "trajectory_identity" in row and row["trajectory_identity"] != expected:
            raise ValueError(
                f"task {task_id!r} has conflicting trajectory_identity: "
                f"observed={row['trajectory_identity']!r} expected={expected!r}"
            )
        row["trajectory_identity"] = expected
        row.setdefault(
            "agent_ref",
            {"type": "responses_api_agents", "name": agent_name},
        )
        annotated.append(row)
    return annotated


# Backward-compatible import name; emitted rows use the generic contract.
annotate_exact_trace_rows = annotate_trajectory_rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL without exposing a partially written training manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--domain", action="append", dest="domains")
    parser.add_argument("--seed", default="osworld-exact-trace-v1")
    parser.add_argument("--group-prefix", default="osworld")
    parser.add_argument("--agent-name", default="osworld_simple_agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_osworld_rows(
        args.input,
        domains=set(args.domains) if args.domains else None,
    )
    train, validation = split_osworld_rows(
        rows,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    train = annotate_trajectory_rows(
        train,
        group_prefix=args.group_prefix,
        agent_name=args.agent_name,
    )
    validation = annotate_trajectory_rows(
        validation,
        group_prefix=args.group_prefix,
        agent_name=args.agent_name,
    )
    write_jsonl_atomic(args.train_output, train)
    write_jsonl_atomic(args.validation_output, validation)
    print(
        f"OSWORLD_EXACT_TRACE_DATA_OK train={len(train)} "
        f"validation={len(validation)} seed={args.seed!r}"
    )


if __name__ == "__main__":
    main()
