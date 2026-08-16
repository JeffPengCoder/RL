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

from copy import deepcopy
import json
from pathlib import Path

import pytest

from nemo_rl.environments.generation_contract import (
    RUNTIME_GENERATION_CONTRACT_SCHEMA_VERSION,
    build_training_admission_contract,
    canonical_digest,
    stable_id,
)
from nemo_rl.environments.nemo_gym_trace import build_rollout_trace_bundle
from nemo_rl.experience.rollout_traces import build_trace_batch_plan
from nemo_rl.experience.sync_exact_trace import (
    build_exact_trace_pending_identity,
    build_exact_trace_wire_identity,
    summarize_exact_trace_plan,
    validate_exact_trace_committed_meta,
)


_FIXTURE_DIR = Path(__file__).parents[2] / "fixtures" / "context_compaction_traces"


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURE_DIR / name).read_text())


def _rekey(bundle: dict, *, rollout_id: str, group_id: str, reward: float) -> dict:
    result = deepcopy(bundle)
    result["rollout_id"] = rollout_id
    result["group_id"] = group_id
    result["reward"] = reward
    trace_ids: dict[str, str] = {}
    for trace in result["physical_traces"]:
        old_trace_id = trace["trace_id"]
        new_trace_id = f"{rollout_id}:trace-{trace['trace_index']:06d}"
        trace["trace_id"] = new_trace_id
        trace_ids[old_trace_id] = new_trace_id
    for call in result["model_calls"]:
        call["source_rollout_id"] = rollout_id
        call["trace_id"] = trace_ids[call["trace_id"]]
    return result


def _training_admitted(
    bundle: dict,
    *,
    sampling_event_id: str = "sampling-event-a",
    policy_version: str = "sync-policy-step-00000000",
) -> dict:
    result = deepcopy(bundle)
    generation_contract = {
        "generation_contract_id": "gym-generation-contract",
        "sampling_contract_id": "gym-sampling-contract",
        "compaction_policy_id": "gym-compaction-policy",
        "loss_normalization": "global_action_token_mean",
        "training_eligible": False,
        "incomplete_reasons": [
            "exact_tokenizer_identity_not_reported_by_generation_server",
            "exact_chat_template_identity_not_reported_by_generation_server",
            "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
        ],
    }
    result["generation_contract"] = generation_contract
    result["sampling_event_id"] = sampling_event_id
    for call in result["model_calls"]:
        call["generation_contract_id"] = generation_contract["generation_contract_id"]
    definitions = {
        "model": {"generation_policy_version": policy_version},
        "tokenizer": {"vocab": "test"},
        "template": {"template": "test"},
        "processor": {"processor": "test"},
    }
    component_ids = {
        "model_contract_id": stable_id("model-contract", definitions["model"]),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract", definitions["tokenizer"]
        ),
        "template_contract_id": stable_id("template-contract", definitions["template"]),
        "processor_contract_id": stable_id(
            "processor-contract", definitions["processor"]
        ),
    }
    runtime = {
        "schema_version": RUNTIME_GENERATION_CONTRACT_SCHEMA_VERSION,
        **component_ids,
        "runtime_contract_id": stable_id(
            "generation-runtime-contract",
            canonical_digest(component_ids),
        ),
        "component_definitions": definitions,
        "training_eligible": True,
        "incomplete_reasons": [],
    }
    result["training_admission"] = build_training_admission_contract(
        generation_contract,
        runtime,
    )
    return result


def _two_trace_variant(bundle: dict) -> dict:
    """Turn the 3-trace golden fixture into a valid 2-trace authority.

    The fifth prompt is made append-compatible with turn four, so this remains
    a fully validated exact trace rather than a hand-edited row-count stub.
    """
    calls = deepcopy(bundle["model_calls"])
    for call in calls:
        call["rollout_id"] = call.pop("source_rollout_id")
    previous = calls[3]
    last = calls[4]
    last["prompt_token_ids"] = [
        *previous["prompt_token_ids"],
        *previous["sampled_token_ids"],
    ]
    last["media_ids"] = deepcopy(previous["media_ids"])
    last["media_occurrences"] = deepcopy(previous["media_occurrences"])
    last["expected_append_compatible"] = True
    last["context_epoch"] = previous["context_epoch"]
    last["segment_index"] = previous["segment_index"]
    last["segment_id"] = previous["segment_id"]
    last["compaction_event_id"] = None
    media_assets = {media_id: {} for call in calls for media_id in call["media_ids"]}
    return build_rollout_trace_bundle(
        rollout_id=bundle["rollout_id"],
        calls=calls,
        boundary_events=[bundle["physical_traces"][1]["boundary_before"]],
        policy_name=bundle["policy_name"],
        group_id=bundle["group_id"],
        source_row_index=bundle["source_row_index"],
        reward=bundle["reward"],
        media_assets=media_assets,
        generation_contract=bundle["generation_contract"],
        final_policy_decision=bundle["final_policy_decision"],
        lineage_deltas=bundle["lineage_deltas"],
        strict=True,
    )


def _multi_trace_plan() -> tuple[dict, list[dict]]:
    first_rollout = _rekey(
        _two_trace_variant(_fixture("k2_compaction.json")),
        rollout_id="rollout-two-traces-a",
        group_id="shared-group",
        reward=0.0,
    )
    second_rollout = _rekey(
        _fixture("k2_compaction.json"),
        rollout_id="rollout-three-traces-b",
        group_id="shared-group",
        reward=1.0,
    )
    bundles = [
        _training_admitted(first_rollout),
        _training_admitted(second_rollout),
    ]
    plan = build_trace_batch_plan(
        bundles,
        rollout_advantages={
            "rollout-two-traces-a": -1.0,
            "rollout-three-traces-b": 1.0,
        },
        expected_rollouts_per_group=2,
        batch_quantum=8,
        optimizer_step_id="grpo-step-00000001",
        training_admission=True,
    )
    return plan, bundles


def _execution_contexts(bundles: list[dict]) -> list[dict]:
    return [
        {
            "schema_version": 1,
            "execution_id": f"execution-{index}",
            "sampling_event_id": bundle["sampling_event_id"],
            "rollout_id": bundle["rollout_id"],
            "group_id": bundle["group_id"],
        }
        for index, bundle in enumerate(bundles)
    ]


def test_pending_handle_is_retry_stable_and_event_scoped():
    common = {
        "generation_policy_version": "sync-policy-step-00000000",
        "optimizer_step_id": "grpo-step-00000001",
        "logical_rollout_count": 2,
        "group_size": 2,
    }
    first = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        **common,
    )
    retry = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        **common,
    )
    next_event = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-b",
        **common,
    )

    assert first == retry
    assert first["pending_handle"] != next_event["pending_handle"]


def test_physical_count_and_logical_scheduler_increment_stay_distinct():
    plan, bundles = _multi_trace_plan()
    pending = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        generation_policy_version="sync-policy-step-00000000",
        optimizer_step_id="grpo-step-00000001",
        logical_rollout_count=2,
        group_size=2,
    )

    summary = summarize_exact_trace_plan(
        plan,
        pending_identity=pending,
        bundles=bundles,
        execution_contexts=_execution_contexts(bundles),
    )

    assert plan["physical_trace_count"] == 5
    assert plan["padding_row_count"] == 3
    assert summary["total_row_count"] == 8
    assert summary["scheduler_step_increment"] == 2


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"sampling_event_id": "different-event"},
            "sampling-event identity",
        ),
        (
            {"generation_policy_version": "different-policy"},
            "generation-policy version",
        ),
        ({"group_size": 1}, "group size"),
    ],
)
def test_plan_admission_must_match_controller_identity(overrides, match):
    plan, bundles = _multi_trace_plan()
    identity_kwargs = {
        "sampling_event_id": "sampling-event-a",
        "generation_policy_version": "sync-policy-step-00000000",
        "optimizer_step_id": "grpo-step-00000001",
        "logical_rollout_count": 2,
        "group_size": 2,
        **overrides,
    }
    pending = build_exact_trace_pending_identity(**identity_kwargs)

    with pytest.raises(ValueError, match=match):
        summarize_exact_trace_plan(
            plan,
            pending_identity=pending,
            bundles=bundles,
            execution_contexts=_execution_contexts(bundles),
        )


def test_plan_must_be_training_admitted():
    plan, bundles = _multi_trace_plan()
    pending = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        generation_policy_version="sync-policy-step-00000000",
        optimizer_step_id="grpo-step-00000001",
        logical_rollout_count=2,
        group_size=2,
    )
    plan["training_admitted"] = False

    with pytest.raises(ValueError):
        summarize_exact_trace_plan(
            plan,
            pending_identity=pending,
            bundles=bundles,
            execution_contexts=_execution_contexts(bundles),
        )


def test_wire_rows_are_plan_derived_and_joinable():
    plan, bundles = _multi_trace_plan()
    pending = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        generation_policy_version="sync-policy-step-00000000",
        optimizer_step_id="grpo-step-00000001",
        logical_rollout_count=2,
        group_size=2,
    )

    sample_ids, tags, extra = build_exact_trace_wire_identity(
        plan,
        pending_identity=pending,
        execution_ids_by_rollout={
            context["rollout_id"]: context["execution_id"]
            for context in _execution_contexts(bundles)
        },
    )

    assert sample_ids == [f"{plan['plan_id']}:{row_index}" for row_index in range(8)]
    assert [tag["row_kind"] for tag in tags] == [
        "physical_trace",
        "physical_trace",
        "physical_trace",
        "physical_trace",
        "physical_trace",
        "padding",
        "padding",
        "padding",
    ]
    assert [tag["parent_rollout_index"] for tag in tags] == [
        0,
        0,
        1,
        1,
        1,
        -1,
        -1,
        -1,
    ]
    assert extra["logical_rollout_count"] == 2
    assert extra["physical_trace_count"] == 5
    assert (
        extra["training_admission_contract_id"]
        == plan["training_admission_contract_id"]
    )
    assert extra["generation_contract_id"] == plan["generation_contract_id"]
    assert [tag["execution_id"] for tag in tags[:5]] == [
        "execution-0",
        "execution-0",
        "execution-1",
        "execution-1",
        "execution-1",
    ]
    for tag in tags[-3:]:
        assert tag["rollout_id"] is None
        assert tag["group_id"] is None
        assert tag["trace_id"] is None
        assert tag["execution_id"] is None
    validate_exact_trace_committed_meta(
        sample_ids=sample_ids,
        tags=tags,
        extra_info=extra,
        plan=plan,
        pending_identity=pending,
        execution_ids_by_rollout=extra["execution_ids_by_rollout"],
    )

    corrupted = deepcopy(tags)
    corrupted[1]["rollout_id"] = "wrong-rollout"
    with pytest.raises(ValueError, match="row tags"):
        validate_exact_trace_committed_meta(
            sample_ids=sample_ids,
            tags=corrupted,
            extra_info=extra,
            plan=plan,
            pending_identity=pending,
            execution_ids_by_rollout=extra["execution_ids_by_rollout"],
        )


def test_execution_identity_is_observability_only_and_fail_closed():
    plan, bundles = _multi_trace_plan()
    pending = build_exact_trace_pending_identity(
        sampling_event_id="sampling-event-a",
        generation_policy_version="sync-policy-step-00000000",
        optimizer_step_id="grpo-step-00000001",
        logical_rollout_count=2,
        group_size=2,
    )
    original_plan_id = plan["plan_id"]
    contexts = _execution_contexts(bundles)
    first = summarize_exact_trace_plan(
        plan,
        pending_identity=pending,
        bundles=bundles,
        execution_contexts=contexts,
    )
    changed = deepcopy(contexts)
    changed[0]["execution_id"] = "execution-new"
    second = summarize_exact_trace_plan(
        plan,
        pending_identity=pending,
        bundles=bundles,
        execution_contexts=changed,
    )

    assert plan["plan_id"] == original_plan_id
    assert first["pending_identity"] == second["pending_identity"]
    assert first["execution_ids_by_rollout"] != second["execution_ids_by_rollout"]
    wrong_event = deepcopy(contexts)
    wrong_event[0]["sampling_event_id"] = "wrong-event"
    with pytest.raises(ValueError, match="sampling event"):
        summarize_exact_trace_plan(
            plan,
            pending_identity=pending,
            bundles=bundles,
            execution_contexts=wrong_event,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sampling_event_id", ""),
        ("generation_policy_version", ""),
        ("logical_rollout_count", 0),
        ("group_size", 0),
    ],
)
def test_pending_identity_fails_closed_on_malformed_control_fields(field, value):
    kwargs = {
        "sampling_event_id": "sampling-event-a",
        "generation_policy_version": "sync-policy-step-00000000",
        "optimizer_step_id": "grpo-step-00000001",
        "logical_rollout_count": 2,
        "group_size": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_exact_trace_pending_identity(**kwargs)
