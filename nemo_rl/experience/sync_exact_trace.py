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

"""Control-plane identities for sync TQ exact-trace training.

The physical row count is not known until a NeMo-Gym rollout has been
materialized.  Sync exact-trace training therefore uses a two-phase protocol:

1. the rollout actor prepares one immutable physical plan under a deterministic
   pending handle;
2. the controller registers the TQ partition for the plan's exact row count;
3. the actor commits those rows once, using plan-derived sample IDs and tags.

This module is deliberately tensor-free.  It defines only the small control
metadata that may cross the Ray boundary; token tensors remain actor/TQ owned.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, TypedDict

from nemo_rl.experience.rollout_traces import validate_trace_batch_plan


_SYNC_EXACT_PENDING_SCHEMA_VERSION = 1


class ExactTracePendingIdentity(TypedDict):
    """Deterministic identity for one prepared, not-yet-consumed TQ batch."""

    schema_version: int
    pending_handle: str
    sampling_event_id: str
    generation_policy_version: str
    optimizer_step_id: str
    logical_rollout_count: int
    group_size: int


class ExactTracePlanSummary(TypedDict):
    """Bounded plan metadata returned to the controller before registration."""

    pending_identity: ExactTracePendingIdentity
    plan_id: str
    training_admission_contract_id: str
    total_row_count: int
    physical_trace_count: int
    padding_row_count: int
    logical_rollout_count: int
    eligible_action_token_count: int
    scheduler_step_increment: int
    execution_ids_by_rollout: dict[str, str]


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def build_exact_trace_pending_identity(
    *,
    sampling_event_id: str,
    generation_policy_version: str,
    optimizer_step_id: str,
    logical_rollout_count: int,
    group_size: int,
) -> ExactTracePendingIdentity:
    """Mint the retry-stable handle for one controller sampling event.

    The handle intentionally excludes execution IDs.  A retry of the same
    controller-owned sampling decision must address the same pending plan,
    while a new sampling event must never alias it.
    """
    identity_without_handle = {
        "schema_version": _SYNC_EXACT_PENDING_SCHEMA_VERSION,
        "sampling_event_id": _require_nonempty_string(
            sampling_event_id,
            field="sampling_event_id",
        ),
        "generation_policy_version": _require_nonempty_string(
            generation_policy_version,
            field="generation_policy_version",
        ),
        "optimizer_step_id": _require_nonempty_string(
            optimizer_step_id,
            field="optimizer_step_id",
        ),
        "logical_rollout_count": _require_positive_int(
            logical_rollout_count,
            field="logical_rollout_count",
        ),
        "group_size": _require_positive_int(group_size, field="group_size"),
    }
    if (
        identity_without_handle["logical_rollout_count"]
        % identity_without_handle["group_size"]
    ):
        raise ValueError("logical_rollout_count must be divisible by group_size")
    payload = json.dumps(
        identity_without_handle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        **identity_without_handle,
        "pending_handle": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def summarize_exact_trace_plan(
    plan: Mapping[str, Any],
    *,
    pending_identity: ExactTracePendingIdentity,
    bundles: Sequence[Mapping[str, Any]],
    execution_contexts: Sequence[Mapping[str, Any]],
) -> ExactTracePlanSummary:
    """Validate a physical plan and expose its registration/scheduler counts."""
    validate_trace_batch_plan(plan, bundles=bundles)
    if plan["training_admitted"] is not True:
        raise ValueError("Sync exact-trace TQ plan is not admitted for training")
    if plan["optimizer_step_id"] != pending_identity["optimizer_step_id"]:
        raise ValueError("Trace plan optimizer step does not match pending identity")
    if plan["logical_rollout_count"] != pending_identity["logical_rollout_count"]:
        raise ValueError("Trace plan logical count does not match pending identity")
    if plan["expected_rollouts_per_group"] != pending_identity["group_size"]:
        raise ValueError("Trace plan group size does not match pending identity")

    sampling_event_ids = {bundle.get("sampling_event_id") for bundle in bundles}
    if sampling_event_ids != {pending_identity["sampling_event_id"]}:
        raise ValueError(
            "Trace bundles do not match the controller sampling-event identity"
        )
    policy_versions: set[Any] = set()
    admission_ids: set[Any] = set()
    for bundle in bundles:
        admission = bundle.get("training_admission")
        if not isinstance(admission, Mapping):
            raise ValueError("Trace bundle has no NeMo-RL training admission")
        admission_ids.add(admission.get("admission_contract_id"))
        runtime = admission.get("runtime_contract")
        definitions = (
            runtime.get("component_definitions")
            if isinstance(runtime, Mapping)
            else None
        )
        model = definitions.get("model") if isinstance(definitions, Mapping) else None
        policy_versions.add(
            model.get("generation_policy_version")
            if isinstance(model, Mapping)
            else None
        )
    if policy_versions != {pending_identity["generation_policy_version"]}:
        raise ValueError(
            "Trace admissions do not match the controller generation-policy version"
        )
    if admission_ids != {plan["training_admission_contract_id"]}:
        raise ValueError("Trace admission identity does not match the physical plan")
    if len(execution_contexts) != len(bundles):
        raise ValueError(
            "Physical execution context count does not match logical rollouts"
        )
    execution_ids_by_rollout: dict[str, str] = {}
    for bundle, context in zip(bundles, execution_contexts, strict=True):
        if not isinstance(context, Mapping):
            raise ValueError("Exact-trace rollout has no physical execution context")
        rollout_id = _require_nonempty_string(
            bundle.get("rollout_id"),
            field="rollout_id",
        )
        execution_id = _require_nonempty_string(
            context.get("execution_id"),
            field=f"execution_id[{rollout_id!r}]",
        )
        if context.get("rollout_id") != rollout_id:
            raise ValueError(
                "Physical execution context disagrees with its logical rollout"
            )
        if context.get("group_id") != bundle.get("group_id"):
            raise ValueError(
                "Physical execution context disagrees with its comparison group"
            )
        if context.get("sampling_event_id") != pending_identity["sampling_event_id"]:
            raise ValueError(
                "Physical execution context disagrees with the sampling event"
            )
        if rollout_id in execution_ids_by_rollout:
            raise ValueError(f"Duplicate logical rollout identity {rollout_id!r}")
        if execution_id in execution_ids_by_rollout.values():
            raise ValueError(f"Duplicate physical execution identity {execution_id!r}")
        execution_ids_by_rollout[rollout_id] = execution_id
    return {
        "pending_identity": pending_identity,
        "plan_id": str(plan["plan_id"]),
        "training_admission_contract_id": _require_nonempty_string(
            plan["training_admission_contract_id"],
            field="training_admission_contract_id",
        ),
        "total_row_count": int(plan["total_row_count"]),
        "physical_trace_count": int(plan["physical_trace_count"]),
        "padding_row_count": int(plan["padding_row_count"]),
        "logical_rollout_count": int(plan["logical_rollout_count"]),
        "eligible_action_token_count": int(plan["eligible_action_token_count"]),
        "scheduler_step_increment": int(plan["logical_rollout_count"]),
        "execution_ids_by_rollout": execution_ids_by_rollout,
    }


def build_exact_trace_wire_identity(
    plan: Mapping[str, Any],
    *,
    pending_identity: ExactTracePendingIdentity,
    execution_ids_by_rollout: Mapping[str, str],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Build stable sample IDs, row tags, and batch metadata from one plan."""
    validate_trace_batch_plan(plan)
    pending_handle = _require_nonempty_string(
        pending_identity.get("pending_handle"),
        field="pending_handle",
    )
    plan_id = str(plan["plan_id"])
    admission_id = _require_nonempty_string(
        plan.get("training_admission_contract_id"),
        field="training_admission_contract_id",
    )
    expected_rollout_ids = set(plan["rollout_ids"])
    if set(execution_ids_by_rollout) != expected_rollout_ids:
        raise ValueError(
            "Physical execution identities do not cover the trace plan rollouts"
        )
    if len(set(execution_ids_by_rollout.values())) != len(execution_ids_by_rollout):
        raise ValueError("Physical execution identities must be unique")
    for rollout_id, execution_id in execution_ids_by_rollout.items():
        _require_nonempty_string(rollout_id, field="rollout_id")
        _require_nonempty_string(
            execution_id,
            field=f"execution_id[{rollout_id!r}]",
        )
    sample_ids: list[str] = []
    tags: list[dict[str, Any]] = []
    for row_index, row in enumerate(plan["rows"]):
        if row["row_index"] != row_index:
            raise ValueError("Trace plan row ordering changed before TQ commit")
        sample_ids.append(f"{plan_id}:{row_index}")
        tags.append(
            {
                "exact_trace": True,
                "pending_handle": pending_handle,
                "sampling_event_id": pending_identity["sampling_event_id"],
                "generation_policy_version": pending_identity[
                    "generation_policy_version"
                ],
                "plan_id": plan_id,
                "optimizer_step_id": str(plan["optimizer_step_id"]),
                "training_admission_contract_id": admission_id,
                "generation_contract_id": str(plan["generation_contract_id"]),
                "row_index": row_index,
                "row_kind": str(row["row_kind"]),
                "parent_rollout_index": int(row["parent_rollout_index"]),
                "rollout_id": row["rollout_id"],
                "execution_id": (
                    execution_ids_by_rollout[row["rollout_id"]]
                    if row["rollout_id"] is not None
                    else None
                ),
                "group_id": row["group_id"],
                "trace_id": row["trace_id"],
                "eligible_action_token_count": int(row["eligible_token_count"]),
            }
        )
    extra_info = {
        "exact_trace": True,
        "pending_handle": pending_handle,
        "sampling_event_id": pending_identity["sampling_event_id"],
        "generation_policy_version": pending_identity["generation_policy_version"],
        "plan_id": plan_id,
        "optimizer_step_id": str(plan["optimizer_step_id"]),
        "training_admission_contract_id": admission_id,
        "generation_contract_id": str(plan["generation_contract_id"]),
        "execution_ids_by_rollout": dict(execution_ids_by_rollout),
        "logical_rollout_count": int(plan["logical_rollout_count"]),
        "physical_trace_count": int(plan["physical_trace_count"]),
        "padding_row_count": int(plan["padding_row_count"]),
        "eligible_action_token_count": int(plan["eligible_action_token_count"]),
    }
    if len(sample_ids) != int(plan["total_row_count"]):
        raise ValueError("Trace plan row count changed before TQ commit")
    return sample_ids, tags, extra_info


def validate_exact_trace_committed_meta(
    *,
    sample_ids: list[str],
    tags: list[dict[str, Any]] | None,
    extra_info: Mapping[str, Any],
    plan: Mapping[str, Any],
    pending_identity: ExactTracePendingIdentity,
    execution_ids_by_rollout: Mapping[str, str],
) -> None:
    """Cross-check returned KV metadata against the plan-derived authority."""
    expected_ids, expected_tags, expected_extra = build_exact_trace_wire_identity(
        plan,
        pending_identity=pending_identity,
        execution_ids_by_rollout=execution_ids_by_rollout,
    )
    if sample_ids != expected_ids:
        raise ValueError("Committed TQ sample IDs disagree with TraceBatchPlan")
    if tags != expected_tags:
        raise ValueError("Committed TQ row tags disagree with TraceBatchPlan")
    for key, value in expected_extra.items():
        if extra_info.get(key) != value:
            raise ValueError(
                f"Committed TQ metadata field {key!r} disagrees with TraceBatchPlan"
            )
