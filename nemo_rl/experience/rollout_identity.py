# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controller-owned identities for logical NeMo-Gym sampling events."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from typing import Any

from nemo_rl.environments.generation_contract import stable_id


_PURPOSE_COMPONENT_RE = re.compile(r"[^a-z0-9]+")
_LEGACY_V2_IDENTITY_FIELDS = (
    "context_compaction_contract_version",
    "context_compaction_rollout_id",
    "context_compaction_group_id",
    "context_compaction_task_id",
    "context_compaction_rollout_index",
    "context_compaction_attempt_index",
)


def new_sampling_event_id(*, purpose: str, step: int | None = None) -> str:
    """Return a globally unique ID for one controller sampling decision.

    A retry of that decision must reuse the returned ID. A later sampling
    decision, even for the same dataset row and policy step, must allocate a new
    one. ``purpose`` and ``step`` are human-readable diagnostics only; UUID
    entropy is the uniqueness authority.
    """
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("sampling event purpose must be a non-empty string")
    normalized_purpose = _PURPOSE_COMPONENT_RE.sub("-", purpose.lower()).strip("-")
    if not normalized_purpose:
        raise ValueError(
            "sampling event purpose must contain an alphanumeric character"
        )
    if step is not None and (
        isinstance(step, bool) or not isinstance(step, int) or step < 0
    ):
        raise ValueError("sampling event step must be a non-negative integer")
    step_component = f"-step-{step:08d}" if step is not None else ""
    return f"sampling-{normalized_purpose}{step_component}-{uuid.uuid4().hex}"


def event_group_id(*, sampling_event_id: str, source_group_id: str) -> str:
    """Derive the comparison group inside one sampling event."""
    _require_non_empty_string(sampling_event_id, "sampling_event_id")
    _require_non_empty_string(source_group_id, "source_group_id")
    return stable_id(
        "group",
        {
            "sampling_event_id": sampling_event_id,
            "source_group_id": source_group_id,
        },
    )


def logical_rollout_id(
    *,
    sampling_event_id: str,
    source_group_id: str,
    task_id: str,
    rollout_index: int,
    attempt_index: int,
) -> str:
    """Derive a retry-stable logical rollout ID inside a sampling event."""
    _require_non_empty_string(sampling_event_id, "sampling_event_id")
    _require_non_empty_string(source_group_id, "source_group_id")
    _require_non_empty_string(task_id, "task_id")
    _require_non_negative_int(rollout_index, "rollout_index")
    _require_non_negative_int(attempt_index, "attempt_index")
    return stable_id(
        "rollout",
        {
            "sampling_event_id": sampling_event_id,
            "source_group_id": source_group_id,
            "task_id": task_id,
            "rollout_index": rollout_index,
            "attempt_index": attempt_index,
        },
    )


def scope_trajectory_identity(
    identity: Mapping[str, Any], *, sampling_event_id: str
) -> dict[str, Any]:
    """Bind one static dataset identity to a controller sampling event.

    Reapplying the same event is idempotent, which is required for stream
    retries. Applying a different event to an already-scoped row fails loudly;
    callers must begin a new event from an unscoped dataset copy instead of
    silently reusing stale request state.
    """
    if identity.get("schema_version") != 1:
        raise ValueError("Unsupported trajectory_identity schema_version")

    observed_event_id = identity.get("sampling_event_id")
    if observed_event_id is not None and observed_event_id != sampling_event_id:
        raise ValueError(
            "trajectory_identity is already bound to a different sampling event: "
            f"observed={observed_event_id!r}, requested={sampling_event_id!r}"
        )

    source_group_id = identity.get("source_group_id")
    if source_group_id is None:
        source_group_id = identity.get("group_id")
    _require_non_empty_string(source_group_id, "source_group_id")

    task_id = identity.get("task_id")
    rollout_index = identity.get("rollout_index")
    attempt_index = identity.get("attempt_index")
    _require_non_empty_string(task_id, "task_id")
    _require_non_negative_int(rollout_index, "rollout_index")
    _require_non_negative_int(attempt_index, "attempt_index")

    scoped_group_id = event_group_id(
        sampling_event_id=sampling_event_id,
        source_group_id=source_group_id,
    )
    scoped_rollout_id = logical_rollout_id(
        sampling_event_id=sampling_event_id,
        source_group_id=source_group_id,
        task_id=task_id,
        rollout_index=rollout_index,
        attempt_index=attempt_index,
    )

    if observed_event_id is not None:
        if identity.get("group_id") != scoped_group_id:
            raise ValueError("Scoped trajectory_identity has an invalid group_id")
        if identity.get("rollout_id") != scoped_rollout_id:
            raise ValueError("Scoped trajectory_identity has an invalid rollout_id")

    scoped = dict(identity)
    scoped.update(
        {
            "sampling_event_id": sampling_event_id,
            "source_group_id": source_group_id,
            "group_id": scoped_group_id,
            "rollout_id": scoped_rollout_id,
        }
    )
    return scoped


def scope_trajectory_identities(
    rows: list[dict[str, Any]], *, sampling_event_id: str
) -> None:
    """Scope every controller-visible trajectory identity in ``rows`` in place.

    Legacy version-2 context-compaction input remains accepted, but is
    normalized to the generic identity before dispatch. This lets old datasets
    participate in event-scoped grouping without extending the legacy wire
    contract with another set of identity fields. Legacy version 1 is left
    alone because its actor-batch-scoped ID is a different compatibility mode.
    """
    _require_non_empty_string(sampling_event_id, "sampling_event_id")
    observed_rollout_ids: set[str] = set()
    for row in rows:
        identity = row.get("trajectory_identity")
        legacy_v2 = (
            identity is None and row.get("context_compaction_contract_version") == 2
        )
        if legacy_v2:
            identity = _normalize_legacy_v2_identity(row)
        if identity is None:
            continue
        if not isinstance(identity, Mapping):
            raise TypeError("trajectory_identity must be a mapping")
        scoped = scope_trajectory_identity(
            identity,
            sampling_event_id=sampling_event_id,
        )
        rollout_id = scoped["rollout_id"]
        if rollout_id in observed_rollout_ids:
            raise ValueError(f"Duplicate logical rollout ID {rollout_id!r}")
        observed_rollout_ids.add(rollout_id)
        if legacy_v2:
            for field in _LEGACY_V2_IDENTITY_FIELDS:
                row.pop(field, None)
        row["trajectory_identity"] = scoped


def _normalize_legacy_v2_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Upgrade one legacy-v2 dataset row to the generic semantic identity."""
    group_id = row.get("context_compaction_group_id")
    task_id = row.get("context_compaction_task_id")
    rollout_index = row.get("context_compaction_rollout_index")
    attempt_index = row.get("context_compaction_attempt_index")
    _require_non_empty_string(group_id, "context_compaction_group_id")
    _require_non_empty_string(task_id, "context_compaction_task_id")
    _require_non_negative_int(
        rollout_index,
        "context_compaction_rollout_index",
    )
    _require_non_negative_int(
        attempt_index,
        "context_compaction_attempt_index",
    )

    # The old rollout_id was static across sampling events. It is deliberately
    # not trusted; scope_trajectory_identity derives the new ID from the
    # controller-owned event plus this source identity. The caller removes the
    # old fields only after the generic identity validates successfully.
    return {
        "schema_version": 1,
        "group_id": group_id,
        "task_id": task_id,
        "rollout_index": rollout_index,
        "attempt_index": attempt_index,
    }


def _require_non_empty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _require_non_negative_int(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
