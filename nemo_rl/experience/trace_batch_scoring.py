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

"""Prepare exact physical trace rows for policy/reference logprob scoring."""

from __future__ import annotations

import math
from typing import Any, Mapping, NotRequired, Protocol, TypedDict

import torch

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_traces import (
    TraceBatchPlan,
    build_trace_batch_plan,
)
from nemo_rl.experience.trace_batch_materialization import (
    TraceBatchMaterialization,
    materialize_trace_batch_plan,
)


class TraceScoringPreparation(TypedDict):
    """Logical GRPO ownership plus exact pre-scoring physical rows."""

    rollout_advantages: dict[str, float]
    plan: TraceBatchPlan
    materialization: TraceBatchMaterialization
    logprob_data: BatchedDataDict[Any]


class TraceScoringResult(TypedDict):
    """Exact physical rows after policy/reference logprob attachment."""

    preparation: TraceScoringPreparation
    train_data: BatchedDataDict[Any]
    rollout_sequence_mask_metrics: NotRequired[dict[str, float]]


class _TraceLogprobPolicy(Protocol):
    def get_logprobs(
        self,
        data: BatchedDataDict[Any],
        timer: Any | None = None,
    ) -> Mapping[str, Any]: ...

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[Any],
        timer: Any | None = None,
    ) -> Mapping[str, Any]: ...


def _require_rollout_aligned_sequence(
    rollout_batch: Mapping[str, Any],
    key: str,
    *,
    rollout_count: int,
) -> list[Any]:
    value = rollout_batch.get(key)
    if not isinstance(value, list) or len(value) != rollout_count:
        raise ValueError(
            f"Trace-aware rollout batch field {key!r} must contain exactly "
            f"{rollout_count} rollout-aligned values"
        )
    return value


def _logical_rollout_sample_masks(
    rollout_batch: Mapping[str, Any],
    *,
    bundles: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Resolve rollout-level loss eligibility without changing GRPO statistics."""
    rollout_count = len(bundles)
    sample_masks = [1.0] * rollout_count
    for key in ("loss_multiplier", "mask_sample", "truncated"):
        value = rollout_batch.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.shape[0] != rollout_count:
                raise ValueError(
                    f"Trace-aware rollout batch field {key!r} is not rollout-aligned"
                )
            values = value.tolist()
        elif isinstance(value, list) and len(value) == rollout_count:
            values = value
        else:
            raise ValueError(
                f"Trace-aware rollout batch field {key!r} is not rollout-aligned"
            )
        if key == "loss_multiplier":
            for index, item in enumerate(values):
                sample_mask = _finite_binary_mask(
                    item,
                    field=f"{key}[{index}]",
                )
                sample_masks[index] *= sample_mask
        else:
            for index, item in enumerate(values):
                if bool(item):
                    sample_masks[index] = 0.0

    result: dict[str, float] = {}
    for bundle, sample_mask in zip(bundles, sample_masks):
        rollout_id = bundle.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError("Trace bundle has no rollout identity")
        if rollout_id in result:
            raise ValueError(f"Duplicate logical rollout ID {rollout_id!r}")
        result[rollout_id] = sample_mask
    if not any(sample_mask == 1.0 for sample_mask in result.values()):
        raise ValueError("Trace-aware GRPO batch has no trainable logical rollout")
    return result


def _finite_binary_mask(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Trace-aware {field} must be a finite binary number")
    result = float(value)
    if not math.isfinite(result) or result not in {0.0, 1.0}:
        raise ValueError(f"Trace-aware {field} must be either 0 or 1")
    return result


def _validate_prompt_group_partition(
    prompt_ids: torch.Tensor,
    bundles: list[Mapping[str, Any]],
) -> None:
    """Prove prompt-token grouping and declared comparison groups agree."""
    for left in range(len(bundles)):
        for right in range(left + 1, len(bundles)):
            same_prompt = torch.equal(prompt_ids[left], prompt_ids[right])
            same_group = bundles[left].get("group_id") == bundles[right].get("group_id")
            if same_prompt != same_group:
                raise ValueError(
                    "Prompt-token equality and rollout comparison-group ownership "
                    f"disagree for rows {left} and {right}"
                )


def _compute_rollout_advantages(
    advantage_estimator: GRPOAdvantageEstimator,
    *,
    bundles: list[Mapping[str, Any]],
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
) -> dict[str, float]:
    if not isinstance(advantage_estimator, GRPOAdvantageEstimator):
        raise TypeError(
            "Trace-aware scoring preparation currently supports only the "
            "standard GRPOAdvantageEstimator"
        )
    rollout_count = len(bundles)
    if (
        prompt_ids.ndim != 2
        or prompt_ids.shape[0] != rollout_count
        or rewards.ndim != 1
        or rewards.shape[0] != rollout_count
    ):
        raise ValueError(
            "Trace-aware prompt IDs and rewards must be logical-rollout aligned"
        )
    if not torch.isfinite(rewards).all():
        raise ValueError("Trace-aware rollout rewards must be finite")
    _validate_prompt_group_partition(prompt_ids, bundles)

    scalar_mask = torch.ones(
        (rollout_count, 1),
        dtype=rewards.dtype,
        device=rewards.device,
    )
    advantages = advantage_estimator.compute_advantage(
        prompt_ids=prompt_ids,
        rewards=rewards,
        mask=scalar_mask,
    )
    if (
        not isinstance(advantages, torch.Tensor)
        or advantages.shape != scalar_mask.shape
        or not torch.isfinite(advantages).all()
    ):
        raise ValueError(
            "GRPO did not produce one finite scalar advantage per logical rollout"
        )

    result: dict[str, float] = {}
    for index, bundle in enumerate(bundles):
        rollout_id = bundle.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ValueError(f"Trace bundle {index} has no rollout identity")
        if rollout_id in result:
            raise ValueError(f"Duplicate logical rollout ID {rollout_id!r}")
        bundle_reward = bundle.get("reward")
        if (
            isinstance(bundle_reward, bool)
            or not isinstance(bundle_reward, (int, float))
            or not math.isclose(
                float(bundle_reward),
                float(rewards[index].item()),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"Trace bundle {rollout_id!r} reward disagrees with GRPO reward"
            )
        result[rollout_id] = float(advantages[index, 0].item())
    return result


def _build_logprob_data(
    materialization: TraceBatchMaterialization,
) -> BatchedDataDict[Any]:
    train_data = materialization["train_data"]
    logprob_data = BatchedDataDict(
        {
            "input_ids": train_data["input_ids"],
            "input_lengths": train_data["input_lengths"],
            "token_mask": train_data["token_mask"],
            "sample_mask": train_data["sample_mask"],
        }
    )
    logprob_data.update(train_data.get_multimodal_dict(as_tensors=False))
    if "routed_experts" in train_data:
        logprob_data["routed_experts"] = train_data["routed_experts"]
    return logprob_data


def prepare_trace_batch_for_scoring(
    rollout_batch: Mapping[str, Any],
    *,
    prompt_ids: torch.Tensor,
    advantage_estimator: GRPOAdvantageEstimator,
    expected_rollouts_per_group: int,
    batch_quantum: int,
    optimizer_step_id: str,
    pad_token_id: int,
    make_sequence_length_divisible_by: int = 1,
    training_admission: bool = False,
    rollout_sequence_mask_ratio_min: float | None = None,
    rollout_sequence_mask_ratio_max: float | None = None,
) -> TraceScoringPreparation:
    """Compute logical GRPO advantages, then expand exact physical rows.

    This function deliberately stops before calling a policy/reference worker.
    Masked logical rollouts remain in their complete comparison group for GRPO
    statistics, while every physical row they own receives ``sample_mask=0``.
    Non-binary weighting, reward rewriting, and non-standard advantage
    estimators remain fail-closed.
    """
    raw_bundles = rollout_batch.get("rollout_trace_bundle")
    if not isinstance(raw_bundles, list) or not raw_bundles:
        raise ValueError(
            "Trace-aware scoring requires rollout_trace_bundle for every rollout"
        )
    if any(not isinstance(bundle, Mapping) for bundle in raw_bundles):
        raise TypeError("rollout_trace_bundle values must be mappings")
    bundles = list(raw_bundles)
    rollout_count = len(bundles)
    physical_message_logs = _require_rollout_aligned_sequence(
        rollout_batch,
        "physical_message_logs",
        rollout_count=rollout_count,
    )
    rollout_sample_masks = _logical_rollout_sample_masks(
        rollout_batch,
        bundles=bundles,
    )

    rewards = rollout_batch.get("total_reward")
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("Trace-aware scoring requires tensor total_reward")
    rollout_advantages = _compute_rollout_advantages(
        advantage_estimator,
        bundles=bundles,
        prompt_ids=prompt_ids,
        rewards=rewards,
    )

    plan = build_trace_batch_plan(
        bundles,
        rollout_advantages=rollout_advantages,
        rollout_sample_masks=rollout_sample_masks,
        expected_rollouts_per_group=expected_rollouts_per_group,
        batch_quantum=batch_quantum,
        optimizer_step_id=optimizer_step_id,
        training_admission=training_admission,
        advantage_estimator_name="grpo",
        sequence_level_ratios_enabled=False,
        rollout_sequence_mask_ratio_min=rollout_sequence_mask_ratio_min,
        rollout_sequence_mask_ratio_max=rollout_sequence_mask_ratio_max,
    )
    physical_message_logs_by_rollout = {
        str(bundle["rollout_id"]): logs
        for bundle, logs in zip(bundles, physical_message_logs)
    }
    if len(physical_message_logs_by_rollout) != rollout_count:
        raise ValueError("Duplicate rollout identity changed trace-log ownership")
    materialization = materialize_trace_batch_plan(
        plan,
        bundles=bundles,
        physical_message_logs_by_rollout=physical_message_logs_by_rollout,
        pad_token_id=pad_token_id,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    return {
        "rollout_advantages": rollout_advantages,
        "plan": plan,
        "materialization": materialization,
        "logprob_data": _build_logprob_data(materialization),
    }


def _validated_logprobs(
    output: Mapping[str, Any],
    *,
    key: str,
    expected_shape: torch.Size,
    effective_token_mask: torch.Tensor,
) -> torch.Tensor:
    value = output.get(key)
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != expected_shape
    ):
        raise ValueError(
            f"Trace-aware worker output {key!r} must be a floating tensor with "
            f"shape {tuple(expected_shape)}"
        )
    if value.device != effective_token_mask.device:
        raise ValueError(
            f"Trace-aware worker output {key!r} is on {value.device}, expected "
            f"{effective_token_mask.device}"
        )
    if not torch.isfinite(value[effective_token_mask]).all():
        raise ValueError(
            f"Trace-aware worker output {key!r} is non-finite on an eligible token"
        )
    # Prompt and padding positions are outside the supported token-level
    # objective. Canonicalize them to zero so masked NaN/Inf values cannot leak
    # through a later multiplication.
    return torch.where(effective_token_mask, value, torch.zeros_like(value))


def apply_rollout_sequence_mask(
    preparation: TraceScoringPreparation,
) -> dict[str, float]:
    """Apply one importance-ratio decision to every row of a logical rollout.

    Context compaction may split one rollout across several physical rows.  The
    mask therefore aggregates all eligible action tokens owned by the logical
    ``rollout_id`` before deciding whether any of its rows may train.  Token-level
    importance-sampling correction remains unchanged in the loss.
    """
    plan = preparation["plan"]
    enabled = bool(plan["sequence_level_clipping_enabled"])
    logical_rollouts = int(plan["logical_rollout_count"])
    pre_masked_rollouts = int(plan["masked_logical_rollout_count"])
    pre_masked_physical_rows = int(plan["masked_physical_trace_count"])
    metrics = {
        "rollout_sequence_mask/enabled": float(enabled),
        "rollout_sequence_mask/logical_rollouts": float(logical_rollouts),
        "rollout_sequence_mask/pre_masked_rollouts": float(pre_masked_rollouts),
        "rollout_sequence_mask/kept_rollouts": float(
            logical_rollouts - pre_masked_rollouts
        ),
        "rollout_sequence_mask/masked_rollouts": float(pre_masked_rollouts),
        "rollout_sequence_mask/masked_physical_rows": float(pre_masked_physical_rows),
        "rollout_sequence_mask/ratio_min": 1.0,
        "rollout_sequence_mask/ratio_mean": 1.0,
        "rollout_sequence_mask/ratio_max": 1.0,
    }
    if not enabled:
        return metrics

    ratio_min = plan["rollout_sequence_mask_ratio_min"]
    ratio_max = plan["rollout_sequence_mask_ratio_max"]
    if not isinstance(ratio_min, float) or not isinstance(ratio_max, float):
        raise ValueError("Enabled rollout sequence mask has no finite bounds")

    train_data = preparation["materialization"]["train_data"]
    required = (
        "token_mask",
        "sample_mask",
        "prev_logprobs",
        "generation_logprobs",
    )
    if any(not isinstance(train_data.get(key), torch.Tensor) for key in required):
        raise TypeError("Rollout sequence mask requires tensor logprob and mask fields")
    token_mask = train_data["token_mask"].bool()
    sample_mask = train_data["sample_mask"]
    prev_logprobs = train_data["prev_logprobs"]
    generation_logprobs = train_data["generation_logprobs"]
    if (
        token_mask.shape != prev_logprobs.shape
        or generation_logprobs.shape != prev_logprobs.shape
        or sample_mask.shape != (prev_logprobs.shape[0],)
    ):
        raise ValueError("Rollout sequence mask fields have inconsistent shapes")

    updated_sample_mask = sample_mask.clone()
    ratios: list[torch.Tensor] = []
    kept_rollouts = 0
    masked_physical_rows = pre_masked_physical_rows
    for rollout_index, row_indices in enumerate(plan["rollout_to_rows"]):
        if not row_indices:
            raise ValueError("Rollout sequence mask found a rollout with no rows")
        row_index = torch.tensor(
            row_indices,
            dtype=torch.int64,
            device=prev_logprobs.device,
        )
        eligible = token_mask.index_select(0, row_index) & (
            sample_mask.index_select(0, row_index).bool().unsqueeze(-1)
        )
        eligible_count = int(torch.count_nonzero(eligible).item())
        expected_count = sum(
            int(plan["rows"][index]["eligible_token_count"]) for index in row_indices
        )
        if expected_count == 0:
            if (
                eligible_count != 0
                or torch.count_nonzero(sample_mask.index_select(0, row_index)).item()
                != 0
            ):
                raise ValueError("Pre-masked rollout changed its planned sample mask")
            continue
        if eligible_count != expected_count:
            raise ValueError(
                "Rollout sequence mask disagrees with planned eligible-token ownership"
            )
        log_ratio = (
            prev_logprobs.index_select(0, row_index)
            - generation_logprobs.index_select(0, row_index)
        )[eligible].mean()
        ratio = torch.exp(log_ratio).detach()
        if not torch.isfinite(ratio):
            raise ValueError(
                f"Rollout {plan['rollout_ids'][rollout_index]!r} has a non-finite "
                "sequence importance ratio"
            )
        ratios.append(ratio)
        keep = bool((ratio >= ratio_min).item() and (ratio <= ratio_max).item())
        if keep:
            kept_rollouts += 1
        else:
            updated_sample_mask[row_index] = 0
            masked_physical_rows += len(row_indices)

    ratio_tensor = torch.stack(ratios).float()
    train_data["sample_mask"] = updated_sample_mask
    metrics.update(
        {
            "rollout_sequence_mask/kept_rollouts": float(kept_rollouts),
            "rollout_sequence_mask/masked_rollouts": float(
                logical_rollouts - kept_rollouts
            ),
            "rollout_sequence_mask/masked_physical_rows": float(masked_physical_rows),
            "rollout_sequence_mask/ratio_min": float(ratio_tensor.min().item()),
            "rollout_sequence_mask/ratio_mean": float(ratio_tensor.mean().item()),
            "rollout_sequence_mask/ratio_max": float(ratio_tensor.max().item()),
        }
    )
    return metrics


def attach_precomputed_trace_logprobs(
    preparation: TraceScoringPreparation,
    *,
    policy_output: Mapping[str, Any] | None,
    reference_output: Mapping[str, Any] | None,
    skip_policy_logprobs: bool = False,
    skip_reference_logprobs: bool = False,
) -> TraceScoringResult:
    """Validate and attach logprobs transported outside the legacy driver.

    Legacy workers return these columns directly. TQ workers write the same
    columns under physical-row IDs and the rollout actor reads them back for
    validation. Sharing this function keeps both transports on one exact-trace
    mask/alignment contract.
    """
    train_data = preparation["materialization"]["train_data"]
    expected_shape = train_data["input_ids"].shape
    effective_token_mask = train_data["token_mask"].bool() & (
        train_data["sample_mask"].bool().unsqueeze(-1)
    )
    if (
        effective_token_mask.shape != expected_shape
        or torch.count_nonzero(effective_token_mask).item()
        != preparation["plan"]["eligible_action_token_count"]
    ):
        raise ValueError(
            "Trace-aware scoring mask disagrees with the physical trace plan"
        )

    if skip_policy_logprobs:
        if policy_output is not None:
            raise ValueError("Skipped policy logprobs must not provide worker output")
        prev_logprobs = torch.zeros_like(train_data["generation_logprobs"])
    else:
        if not isinstance(policy_output, Mapping):
            raise TypeError("Policy logprob worker output must be a mapping")
        prev_logprobs = _validated_logprobs(
            policy_output,
            key="logprobs",
            expected_shape=expected_shape,
            effective_token_mask=effective_token_mask,
        )

    if skip_reference_logprobs:
        if reference_output is not None:
            raise ValueError(
                "Skipped reference logprobs must not provide worker output"
            )
        reference_logprobs = torch.zeros_like(prev_logprobs)
    else:
        if not isinstance(reference_output, Mapping):
            raise TypeError("Reference logprob worker output must be a mapping")
        reference_logprobs = _validated_logprobs(
            reference_output,
            key="reference_logprobs",
            expected_shape=expected_shape,
            effective_token_mask=effective_token_mask,
        )

    train_data["prev_logprobs"] = prev_logprobs
    train_data["reference_policy_logprobs"] = reference_logprobs
    rollout_sequence_mask_metrics = apply_rollout_sequence_mask(preparation)
    return {
        "preparation": preparation,
        "train_data": train_data,
        "rollout_sequence_mask_metrics": rollout_sequence_mask_metrics,
    }


def score_prepared_trace_batch(
    preparation: TraceScoringPreparation,
    *,
    policy: _TraceLogprobPolicy,
    timer: Any | None = None,
    skip_policy_logprobs: bool = False,
    skip_reference_logprobs: bool = False,
) -> TraceScoringResult:
    """Call logprob workers on exact rows and validate their returned alignment.

    The caller remains responsible for worker mode transitions. This function
    does not compute ratios, loss, gradients, or optimizer/scheduler state.
    """
    logprob_data = preparation["logprob_data"]
    policy_output: Mapping[str, Any] | None = None
    if skip_policy_logprobs:
        pass
    else:
        policy_output = policy.get_logprobs(logprob_data, timer=timer)

    reference_output: Mapping[str, Any] | None = None
    if skip_reference_logprobs:
        pass
    else:
        reference_output = policy.get_reference_policy_logprobs(
            logprob_data,
            timer=timer,
        )

    return attach_precomputed_trace_logprobs(
        preparation,
        policy_output=policy_output,
        reference_output=reference_output,
        skip_policy_logprobs=skip_policy_logprobs,
        skip_reference_logprobs=skip_reference_logprobs,
    )
