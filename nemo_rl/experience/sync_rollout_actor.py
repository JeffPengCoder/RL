# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Sync GRPO rollout actor — sibling of ``async_utils``.

Houses :class:`SyncRolloutActor`, the Ray actor that owns the multi-turn
rollout loop AND the post-rollout flatten / mask / prompt extraction /
reward shaping / baseline-std for a sync GRPO step. The driver dispatches
a per-step prompt batch + uids; the actor runs ``run_multi_turn_rollout``
(or async / nemo_gym variants), then writes the bulk schema to TQ via
:func:`nemo_rl.data_plane.column_io.kv_first_write`. Only a ``KVBatchMeta``
and a small per-sample ``driver_carry`` dict (rewards, masks, lengths,
baseline/std, prompt_ids_for_adv) cross back to the driver via Ray.

**Goal — rollout 1-hop put**: bulk tensors (input_ids, output_ids,
attention_mask, position_ids, multi_modal_inputs, generation_logprobs,
token_mask) stay actor-side until ``put_samples``, then live only in
TQ. Driver never holds these bytes between rollout finish and train
fan-out.

The actor is the sync counterpart to
:class:`nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector`. It
intentionally does not buffer or stream — sync GRPO consumes the whole
step batch in one call.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.column_io import kv_first_write, read_columns, write_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.packed_tensor_wire import (
    PACKED_TENSOR_WIRE_SCHEMA_KEY,
    describe_packed_tensor_wire,
    packed_tensor_schema_from_extra_info,
)
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    get_nemo_gym_thinking_tags,
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
    run_nemo_gym_rollout_sync,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.utils.logger import should_log_nemo_gym_full_result_tables
from nemo_rl.utils.r3_trace import trace_rollout_payload

# Carry keys producible by the rollout actor only when the caller opts in.
# These are np.ndarray(object) per-row arrays from decompose_message_log; the
# default driver_carry omits them because BatchedDataDict.select_indices on
# the training/dynamic-sampling path only handles tensors/lists. Validation
# requests them explicitly to print per-sample message logs.
OPT_IN_CARRY_KEYS: tuple[str, ...] = ("turn_roles", "turn_contents")


def _flatten_rollout_message_log_for_tq(
    message_logs: list[Any],
    prompt_lengths: torch.Tensor,
    *,
    pad_token_id: int,
    make_sequence_length_divisible_by: int,
) -> tuple[BatchedDataDict[Any], torch.Tensor, BatchedDataDict[Any]]:
    """Prepare rollout message logs for the TQ payload and driver carry."""
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
    from nemo_rl.experience.rollouts import backfill_missing_routed_experts

    pad = {"pad_value_dict": {"token_ids": pad_token_id}}
    # Must precede the prompt extraction: it reuses the same message dicts, so
    # backfilling here also covers the prompt flatten below.
    backfill_missing_routed_experts(message_logs)
    prompt_message_logs = extract_initial_prompt_messages(
        message_logs,
        prompt_lengths,
    )
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        **pad,
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        **pad,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    return flat, input_lengths, prompt_flat


@ray.remote  # pragma: no cover
class SyncRolloutActor:
    """Per-step rollout dispatcher.

    Runs: rollout + flatten + mask + prompt extraction + baseline/std + TQ put.
    Returns ``(meta, driver_carry, rollout_metrics, gen_metrics)``.

    Lifecycle: one instance per ``grpo_train_sync`` invocation. The driver
    instantiates with the same handles it would normally pass to
    ``run_multi_turn_rollout`` plus the data-plane config so the actor
    can attach as a TQ client (``bootstrap=False`` — controller is
    bootstrapped on the driver via ``TQPolicy``).
    """

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: Any,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: Any,
        dp_cfg: dict[str, Any],
    ) -> None:
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config

        from nemo_rl.data_plane import build_data_plane_client

        self._dp_client = build_data_plane_client(dp_cfg, bootstrap=False)
        # Exact-trace uses a two-phase prepare/register/commit protocol because
        # its physical row count is unknown until rollout materialization. One
        # actor owns at most one plan, and retains it after commit so a lost RPC
        # response can be retried without another rollout or duplicate put.
        self._exact_trace_pending: dict[str, Any] | None = None
        self._exact_trace_failed_identity: dict[str, Any] | None = None

    def rollout_to_tq(
        self,
        input_batch: BatchedDataDict[Any],
        *,
        partition_id: str,
        generation_only: bool,
        generation_policy_version: str | None,
        sampling_event_id: str,
        group_size: int = 1,
        first_iter: bool = True,
        finish_generation: bool = True,
        task_to_env_override: Optional[dict[str, EnvironmentInterface]] = None,
        carry_keys: Optional[list[str]] = None,
    ) -> tuple[
        KVBatchMeta,
        dict[str, Any],
        dict[str, Any],
        Optional[dict[str, Any]],
    ]:
        """Run the full per-step generation cycle and write bulk data to TQ.

        Bundles six steps into one Ray round-trip so the driver only sees
        a single RPC instead of separate calls for each:

        1. **Reset metrics** — ``policy_generation.clear_logger_metrics()``
           clears per-step generation accumulators before the rollout.
        2. **Rollout** — runs ``run_multi_turn_rollout`` (or the async /
           nemo-gym variants) to produce ``final_batch``.
        3. **Flatten + mask + prompt extraction** — converts
           ``message_log`` layout to flat tensors; builds token mask,
           sample mask, prompt-only ids, baseline/std.
        4. **Write bulk to TQ** — ``kv_first_write`` puts every tensor
           field in one flat ``put_samples``; the driver never touches
           bulk bytes.
        5. **Release GPU** — ``policy_generation.finish_generation()``
           frees KV cache and inference state so the trainer can use the
           GPU immediately.
        6. **Capture metrics** — ``policy_generation.get_logger_metrics()``
           collects generation stats (throughput, etc.) and returns them
           to the driver in the result tuple.

        The driver receives ``(meta, driver_carry, rollout_metrics,
        generation_logger_metrics)`` and uses ``driver_carry`` for its
        own per-row compute (rewards, advantages, dynamic sampling).

        Args:
            input_batch: Per-step prompt batch (already repeat-interleaved).
            partition_id: TQ partition target.
            generation_only: Scheduler-owned rollout intent. Validation must
                pass ``True`` so NeMo-Gym selects its evaluation purpose and
                sampling profile; training must pass ``False``. This is
                required instead of defaulted so new controller call sites
                cannot silently turn evaluation into training traffic.
            generation_policy_version: Controller-owned policy identity for
                exact-trace admission. Validation passes ``None``; training
                passes the synchronized policy step.
            sampling_event_id: Controller-owned identity for this sampling
                decision. A caller retry must reuse the same value.
            group_size: Rollouts per original prompt. One uid is minted
                per prompt; bulk keys are ``f"{uid}_g{i}"`` where ``i``
                ranges over the per-prompt expansion (group × rollout
                turns). Train passes ``num_generations_per_prompt``; val
                passes ``1``.
            first_iter: True on the first DS iteration of a step; drives
                ``policy_generation.snapshot_step_metrics()`` so per-step
                metrics align with the legacy ``grpo.grpo_train`` path.
            finish_generation: Call ``policy_generation.finish_generation()``
                at the tail. Default ``True`` matches the training step
                (one rollout per step, release KV after). Validation sets
                ``False`` so inference state survives across val batches;
                the trainer owns the explicit ``finish_generation()`` call
                at the end of the val pass.
            task_to_env_override: Per-call task → env map. ``None`` uses
                ``self.task_to_env`` (training envs supplied at construction).
                Validation passes ``val_task_to_env`` here so val rollouts
                run against the val env set without rebuilding the actor.
            carry_keys: Names of per-row tensors to return in
                ``driver_carry``. ``None`` returns every available key
                (training uses this). Validation passes a slim list
                (e.g. ``["total_reward"]``) to avoid wasting Ray transfer
                on fields it doesn't consume.

        Returns:
            ``(meta, driver_carry, rollout_metrics, generation_logger_metrics)``
            where ``driver_carry`` is a per-row dict of tensors the driver
            uses for compute (rewards, masks, lengths, prompt_ids_for_adv,
            …) — stays on the driver, never crosses an actor boundary.
        """
        # Lazy imports — avoid pulling grpo into this module at load.
        from nemo_rl.algorithms.grpo import (
            _should_use_async_rollouts,
            _should_use_nemo_gym,
        )
        from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys
        from nemo_rl.data.llm_message_utils import (
            MESSAGE_LOG_BULK_FIELDS,
            decompose_message_log,
        )

        # Per-step generation-side metric hooks: snapshot once on the
        # first DS iter so backends with per-step deltas have a stable
        # anchor; clear accumulators before every rollout. Mirrors
        # legacy ``grpo_train``.
        if self.policy_generation is not None:
            if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
                self.policy_generation.snapshot_step_metrics()
            self.policy_generation.clear_logger_metrics()

        cfg = self.master_config
        task_to_env = (
            task_to_env_override
            if task_to_env_override is not None
            else self.task_to_env
        )
        common = dict(
            policy_generation=self.policy_generation,
            input_batch=input_batch,
            tokenizer=self.tokenizer,
            task_to_env=task_to_env,
            greedy=False,
        )

        # Rollout dispatch (mirrors grpo_sync.py:294-349).
        if _should_use_nemo_gym(cfg):
            r = run_nemo_gym_rollout_sync(
                **common,
                max_seq_len=None,
                max_rollout_turns=None,
                generation_config=cfg.policy["generation"],
                log_full_result_tables=should_log_nemo_gym_full_result_tables(
                    wandb_enabled=cfg.logger["wandb_enabled"],
                    wandb_config=cfg.logger["wandb"],
                ),
                effort_config=EffortLevelsConfig.model_validate(
                    cfg.env["nemo_gym"].get("effort_levels")
                )
                if "nemo_gym" in cfg.env
                and cfg.env["nemo_gym"].get("effort_levels") is not None
                else None,
                reward_penalty_config=cfg.reward_penalties,
                thinking_tags=get_nemo_gym_thinking_tags(cfg.env),
                generation_only=generation_only,
                generation_policy_version=generation_policy_version,
                sampling_event_id=sampling_event_id,
            )
            final_batch, rollout_metrics = r.final_batch, r.rollout_metrics
        else:
            runner = (
                run_async_multi_turn_rollout
                if _should_use_async_rollouts(cfg)
                else run_multi_turn_rollout
            )
            final_batch, rollout_metrics = runner(
                **common,
                max_seq_len=cfg.policy["max_total_sequence_length"],
                max_rollout_turns=cfg.grpo.max_rollout_turns,
            )
        fb = final_batch.to("cpu")
        del final_batch

        # Flatten message_log → bulk tensors + extract original prompt ids.
        # GRPO masks only generated assistant turns, even if the dataset
        # prompt itself contains assistant messages as conversation history.
        flat, input_lengths, prompt_flat = _flatten_rollout_message_log_for_tq(
            fb["message_log"],
            fb["length"],
            pad_token_id=self.tokenizer.pad_token_id,
            make_sequence_length_divisible_by=cfg.policy[
                "make_sequence_length_divisible_by"
            ],
        )

        router_replay_enabled = bool(
            (cfg.policy.get("router_replay") or {}).get("enabled", False)
        )
        if router_replay_enabled and ROUTED_EXPERTS_FIELD not in flat:
            raise RuntimeError(
                "policy.router_replay.enabled=true requires routed_experts in "
                "the rollout bulk payload, but rollout flattening did not "
                "produce that field. Check vLLM routed-expert capture and the "
                "message-log flattening path."
            )

        # TQ bulk payload — DP_TRAIN_FIELDS + multimodal extras.
        bulk_batch = BatchedDataDict[Any](
            {
                "input_ids": flat["token_ids"],
                "input_lengths": input_lengths,
                "generation_logprobs": flat["generation_logprobs"],
                "token_mask": flat["token_loss_mask"],
                "sample_mask": fb["loss_multiplier"],
            }
        )
        if ROUTED_EXPERTS_FIELD in flat:
            bulk_batch[ROUTED_EXPERTS_FIELD] = flat[ROUTED_EXPERTS_FIELD]
        for k, v in flat.get_multimodal_dict(as_tensors=False).items():
            if isinstance(v, (torch.Tensor, PackedTensor)):
                bulk_batch[k] = v
        # ``content`` (raw assistant text per sample) — rides TQ as a
        # NonTensorStack so the driver can fetch it back at jsonl time
        # (kv_first_write wraps it via NonTensorStack).
        if "content" in flat:
            bulk_batch["content"] = np.asarray(flat["content"], dtype=object)

        # Split `message_log` into per-field arrays instead of pickling
        # the list-of-dicts-with-tensors per row. Consumer rebuilds
        # `message_log` on read; external API stays the same.
        decomposed = decompose_message_log(fb["message_log"])
        for k in MESSAGE_LOG_BULK_FIELDS:
            bulk_batch[k] = decomposed[k]

        # Pass through remaining non-tensor fb fields as object arrays;
        # `message_log` is excluded since its tensors live in the
        # decomposed fields above (per-row pickle of dict-with-tensors
        # would smuggle aliased views into the wire).
        for k, v in fb.items():
            if (
                isinstance(v, torch.Tensor)
                or k in bulk_batch
                or k in {"message_log", "rollout_execution_context"}
            ):
                continue
            bulk_batch[k] = (
                v
                if isinstance(v, np.ndarray) and v.dtype == object
                else np.asarray(v, dtype=object)
            )

        # Slice — only what the driver can't derive from a TQ slice fetch
        # (anything containing `message_log` or per-token data would
        # force a fetch). Driver does scale_rewards / reward_shaping /
        # overlong filtering / baseline-std on this slice.
        truncated = fb["truncated"]
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool)
        length = fb.get("length", input_lengths)
        if not isinstance(length, torch.Tensor):
            length = torch.tensor(length)
        driver_carry = {
            "total_reward": fb["total_reward"],
            "loss_multiplier": fb["loss_multiplier"],
            "truncated": truncated,
            "length": length,
            "input_lengths": input_lengths,
            "prompt_ids_for_adv": prompt_flat["token_ids"],
            # Computed by decompose_message_log above; feeds
            # apply_reward_shaping on the driver without a TQ fetch.
            "response_token_lengths": decomposed["response_token_lengths"],
        }
        # GDPO multi-reward components: scale_rewards iterates these
        # keys driver-side and the GDPO advantage estimator reads them
        # from ``adv_inputs``. Plumb them through ``driver_carry``
        # rather than forcing a separate TQ fetch.
        for k in get_gdpo_reward_component_keys(fb):
            driver_carry[k] = fb[k]
        if carry_keys is not None:
            for k in OPT_IN_CARRY_KEYS:
                if k in carry_keys:
                    driver_carry[k] = decomposed[k]
            missing = set(carry_keys) - driver_carry.keys()
            if missing:
                raise KeyError(
                    f"rollout_to_tq: carry_keys {sorted(missing)} not produced; "
                    f"valid keys: {sorted(driver_carry)}"
                )
            driver_carry = {k: driver_carry[k] for k in carry_keys}

        n_samples = int(bulk_batch["sample_mask"].shape[0])
        input_size = int(input_batch.size)
        if group_size <= 0 or input_size % group_size != 0:
            raise ValueError(
                f"input_batch.size={input_size} is not divisible by group_size={group_size}"
            )
        n_prompts = input_size // group_size
        if n_prompts == 0 or n_samples % n_prompts != 0:
            raise ValueError(
                f"bulk_batch has {n_samples} samples; not divisible by n_prompts={n_prompts}"
            )
        n_per_prompt = n_samples // n_prompts
        uids = [str(uuid.uuid4()) for _ in range(n_prompts)]
        sample_ids = [f"{uid}_g{i}" for uid in uids for i in range(n_per_prompt)]
        trace_rollout_payload(keys=sample_ids, data=bulk_batch)
        meta = kv_first_write(
            bulk_batch,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            extra_info={"rollout_metrics": rollout_metrics},
            task_name=partition_id,
            pad_to_multiple=int(
                cfg.policy.get("make_sequence_length_divisible_by") or 1
            ),
        )

        if self.policy_generation is not None:
            if finish_generation:
                self.policy_generation.finish_generation()
            gen_metrics = self.policy_generation.get_logger_metrics()
        else:
            gen_metrics = None
        return meta, BatchedDataDict(driver_carry), rollout_metrics, gen_metrics

    def prepare_pending_exact_trace(
        self,
        input_batch: BatchedDataDict[Any],
        *,
        generation_policy_version: str,
        sampling_event_id: str,
        group_size: int,
        batch_quantum: int,
        optimizer_step_id: str,
        first_iter: bool = True,
        finish_generation: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], Optional[dict[str, Any]]]:
        """Roll out and materialize one exact physical batch without TQ I/O.

        A retry with the same controller identity returns the already prepared
        plan. A different event is rejected while a plan is pending. The
        controller must register ``summary['total_row_count']`` with
        ``group_size=None`` before calling :meth:`commit_pending_exact_trace`.
        """
        from nemo_rl.algorithms.grpo import (
            _context_compaction_sequence_mask_bounds,
            _create_advantage_estimator,
            _should_use_nemo_gym,
            extract_initial_prompt_messages,
        )
        from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt
        from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
        from nemo_rl.experience.rollouts import backfill_missing_routed_experts
        from nemo_rl.experience.sync_exact_trace import (
            build_exact_trace_pending_identity,
            summarize_exact_trace_plan,
        )
        from nemo_rl.experience.trace_batch_scoring import (
            prepare_trace_batch_for_scoring,
        )

        pending_identity = build_exact_trace_pending_identity(
            sampling_event_id=sampling_event_id,
            generation_policy_version=generation_policy_version,
            optimizer_step_id=optimizer_step_id,
            logical_rollout_count=int(input_batch.size),
            group_size=group_size,
        )
        pending = self._exact_trace_pending
        failed_identity = self._exact_trace_failed_identity
        if failed_identity is not None:
            if failed_identity["pending_handle"] == pending_identity["pending_handle"]:
                raise RuntimeError(
                    "The non-idempotent NeMo-Gym /run for this sampling event "
                    "already failed after dispatch; refusing an implicit replay"
                )
            raise RuntimeError(
                "SyncRolloutActor retains a failed exact-trace sampling event; "
                "abort it explicitly before starting a new event"
            )
        if pending is not None:
            if pending["pending_identity"] != pending_identity:
                raise RuntimeError(
                    "SyncRolloutActor already owns a different exact-trace "
                    f"pending plan {pending['pending_identity']['pending_handle']!r}"
                )
            return (
                pending["controller_summary"],
                pending["rollout_metrics"],
                pending["generation_logger_metrics"],
            )

        cfg = self.master_config
        if not _should_use_nemo_gym(cfg):
            raise ValueError("Exact-trace TQ training requires NeMo-Gym rollouts")
        if self.policy_generation is not None:
            if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
                self.policy_generation.snapshot_step_metrics()
            self.policy_generation.clear_logger_metrics()

        rollout_succeeded = False
        try:
            # Publish a fail-closed identity before the first call that may
            # dispatch the non-idempotent Gym /run. It stays live until the
            # fully materialized pending state is published below, covering
            # finish-generation and logger-metric postprocessing failures too.
            self._exact_trace_failed_identity = pending_identity
            result = run_nemo_gym_rollout_sync(
                policy_generation=self.policy_generation,
                input_batch=input_batch,
                tokenizer=self.tokenizer,
                task_to_env=self.task_to_env,
                greedy=False,
                max_seq_len=None,
                max_rollout_turns=None,
                generation_config=cfg.policy["generation"],
                log_full_result_tables=should_log_nemo_gym_full_result_tables(
                    wandb_enabled=cfg.logger["wandb_enabled"],
                    wandb_config=cfg.logger["wandb"],
                ),
                effort_config=EffortLevelsConfig.model_validate(
                    cfg.env["nemo_gym"].get("effort_levels")
                )
                if "nemo_gym" in cfg.env
                and cfg.env["nemo_gym"].get("effort_levels") is not None
                else None,
                reward_penalty_config=cfg.reward_penalties,
                thinking_tags=get_nemo_gym_thinking_tags(cfg.env),
                generation_only=False,
                generation_policy_version=generation_policy_version,
                sampling_event_id=sampling_event_id,
            )
            fb = result.final_batch.to("cpu")
            rollout_metrics = result.rollout_metrics

            # Prompt identity and router replay must be made complete before
            # the exact materializer compares the logical and physical views.
            backfill_missing_routed_experts(fb["message_log"])
            physical_message_logs = fb.get("physical_message_logs")
            if not isinstance(physical_message_logs, list):
                raise TypeError(
                    "Exact-trace rollout did not return physical_message_logs"
                )
            for rollout_logs in physical_message_logs:
                if not isinstance(rollout_logs, list):
                    raise TypeError("Physical message logs are not rollout-aligned")
                backfill_missing_routed_experts(rollout_logs)

            prompt_logs = extract_initial_prompt_messages(
                fb["message_log"],
                fb["length"],
            )
            prompt_flat, _ = batched_message_log_to_flat_message(
                prompt_logs,
                pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
            )
            prompt_ids = prompt_flat["token_ids"]
            sequence_mask_min, sequence_mask_max = (
                _context_compaction_sequence_mask_bounds(cfg)
            )
            preparation = prepare_trace_batch_for_scoring(
                fb,
                prompt_ids=prompt_ids,
                advantage_estimator=_create_advantage_estimator(cfg),
                expected_rollouts_per_group=group_size,
                batch_quantum=batch_quantum,
                optimizer_step_id=optimizer_step_id,
                pad_token_id=self.tokenizer.pad_token_id,
                make_sequence_length_divisible_by=cfg.policy[
                    "make_sequence_length_divisible_by"
                ],
                training_admission=True,
                rollout_sequence_mask_ratio_min=sequence_mask_min,
                rollout_sequence_mask_ratio_max=sequence_mask_max,
            )
            plan = preparation["plan"]
            plan_summary = summarize_exact_trace_plan(
                plan,
                pending_identity=pending_identity,
                bundles=fb["rollout_trace_bundle"],
                execution_contexts=fb["rollout_execution_context"],
            )
            train_data = preparation["materialization"]["train_data"]
            train_data.to("cpu")

            unsupported_fields = [
                key
                for key, value in train_data.items()
                if not isinstance(value, (torch.Tensor, PackedTensor))
            ]
            if unsupported_fields:
                raise TypeError(
                    "Exact-trace TQ payload contains unsupported fields "
                    f"{sorted(unsupported_fields)!r}"
                )
            router_replay_enabled = bool(
                (cfg.policy.get("router_replay") or {}).get("enabled", False)
            )
            if router_replay_enabled and ROUTED_EXPERTS_FIELD not in train_data:
                raise RuntimeError(
                    "policy.router_replay.enabled=true requires routed_experts "
                    "on every exact physical trace row"
                )

            physical_sample_ids = [
                f"{plan['plan_id']}:{row_index}"
                for row_index in range(int(plan["total_row_count"]))
            ]
            packed_tensor_wire_schema = describe_packed_tensor_wire(
                train_data,
                sample_ids=physical_sample_ids,
            )

            rewards = fb["total_reward"]
            baselines, stds = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=cfg.grpo.use_leave_one_out_baseline,
            )
            materialization = preparation["materialization"]
            content = [
                "".join(str(message.get("content", "")) for message in message_log)
                for message_log in materialization["materialized_message_logs"]
            ]
            # Decoded text is potentially large and is logging data, not
            # two-phase control metadata. Store it beside the tensor rows in
            # TQ as an object column; the controller fetches it only when its
            # JSONL logging path needs it.
            train_data["content"] = np.asarray(content, dtype=object)
            controller_summary: dict[str, Any] = {
                **plan_summary,
                "plan": plan,
                "logical_rewards": rewards.tolist(),
                "logical_baselines": baselines.tolist(),
                "logical_stds": stds.tolist(),
                "logical_prompt_lengths": fb["length"].tolist(),
                "rollout_advantages": [
                    preparation["rollout_advantages"][rollout_id]
                    for rollout_id in plan["rollout_ids"]
                ],
                "physical_input_lengths": train_data["input_lengths"].tolist(),
                "row_rewards": materialization["row_rewards"].tolist(),
                "packed_tensor_wire_schema": packed_tensor_wire_schema,
            }
            rollout_succeeded = True
        finally:
            if not rollout_succeeded:
                self._exact_trace_failed_identity = pending_identity
            if self.policy_generation is not None and finish_generation:
                try:
                    self.policy_generation.finish_generation()
                except Exception:
                    self._exact_trace_failed_identity = pending_identity
                    raise

        if not rollout_succeeded:
            raise AssertionError("Exact-trace rollout exited without a result")
        generation_logger_metrics = (
            self.policy_generation.get_logger_metrics()
            if self.policy_generation is not None
            else None
        )
        self._exact_trace_pending = {
            "pending_identity": pending_identity,
            "controller_summary": controller_summary,
            "rollout_metrics": rollout_metrics,
            "generation_logger_metrics": generation_logger_metrics,
            "preparation": preparation,
            "commit_state": "prepared",
            "committed_meta": None,
            "partition_id": None,
            "intended_sample_ids": None,
        }
        self._exact_trace_failed_identity = None
        return controller_summary, rollout_metrics, generation_logger_metrics

    def commit_pending_exact_trace(
        self,
        *,
        pending_handle: str,
        partition_id: str,
    ) -> KVBatchMeta:
        """Commit a prepared physical batch once after controller registration."""
        from nemo_rl.experience.sync_exact_trace import (
            build_exact_trace_wire_identity,
            validate_exact_trace_committed_meta,
        )

        pending = self._require_exact_trace_pending(pending_handle)
        committed_meta = pending["committed_meta"]
        if committed_meta is not None:
            if pending["partition_id"] != partition_id:
                raise ValueError("Exact-trace retry changed its TQ partition")
            return committed_meta
        if pending["commit_state"] == "committing":
            if pending["partition_id"] != partition_id:
                raise ValueError("Exact-trace retry changed its TQ partition")
            raise RuntimeError(
                "Exact-trace TQ commit outcome is ambiguous; abort the pending "
                "handle to clear its plan-derived sample IDs before continuing"
            )
        if pending["commit_state"] != "prepared":
            raise RuntimeError(
                "Exact-trace pending plan has invalid commit state "
                f"{pending['commit_state']!r}"
            )

        preparation = pending["preparation"]
        plan = preparation["plan"]
        sample_ids, tags, extra_info = build_exact_trace_wire_identity(
            plan,
            pending_identity=pending["pending_identity"],
            execution_ids_by_rollout=pending["controller_summary"][
                "execution_ids_by_rollout"
            ],
        )
        packed_tensor_wire_schema = pending["controller_summary"].get(
            "packed_tensor_wire_schema"
        )
        if packed_tensor_wire_schema is not None:
            extra_info[PACKED_TENSOR_WIRE_SCHEMA_KEY] = packed_tensor_wire_schema
        train_data = preparation["materialization"]["train_data"]
        trace_rollout_payload(keys=sample_ids, data=train_data)
        # Record the complete cleanup authority immediately before the first
        # non-idempotent KV write. If the write succeeds but the RPC response
        # is lost, an explicit abort can clear the deterministic keys without
        # replaying either the Gym /run or the Mooncake put.
        pending["commit_state"] = "committing"
        pending["partition_id"] = partition_id
        pending["intended_sample_ids"] = list(sample_ids)
        meta = kv_first_write(
            train_data,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            extra_info=extra_info,
            task_name=partition_id,
            pad_to_multiple=int(
                self.master_config.policy.get("make_sequence_length_divisible_by") or 1
            ),
            tags=tags,
        )
        validate_exact_trace_committed_meta(
            sample_ids=meta.sample_ids,
            tags=meta.tags,
            extra_info=meta.extra_info,
            plan=plan,
            pending_identity=pending["pending_identity"],
            execution_ids_by_rollout=pending["controller_summary"][
                "execution_ids_by_rollout"
            ],
        )
        if (
            packed_tensor_schema_from_extra_info(meta.extra_info)
            != packed_tensor_wire_schema
        ):
            raise ValueError("Committed exact-trace media schema changed")
        pending["committed_meta"] = meta
        pending["commit_state"] = "committed"
        return meta

    def validate_pending_exact_trace_scoring(
        self,
        *,
        pending_handle: str,
        meta: KVBatchMeta,
        skip_policy_logprobs: bool,
        skip_reference_logprobs: bool,
    ) -> dict[str, Any]:
        """Validate TQ worker columns on the actor-owned physical authority."""
        from nemo_rl.experience.sync_exact_trace import (
            validate_exact_trace_committed_meta,
        )
        from nemo_rl.experience.trace_batch_scoring import (
            attach_precomputed_trace_logprobs,
        )

        pending = self._require_exact_trace_pending(pending_handle)
        if pending["commit_state"] != "committed" or pending["committed_meta"] is None:
            raise RuntimeError("Cannot score an uncommitted exact-trace plan")
        committed_meta = pending["committed_meta"]
        if (
            meta.partition_id != committed_meta.partition_id
            or meta.sample_ids != committed_meta.sample_ids
            or meta.sequence_lengths != committed_meta.sequence_lengths
        ):
            raise ValueError("Exact-trace scoring metadata changed after TQ commit")
        preparation = pending["preparation"]
        plan = preparation["plan"]
        validate_exact_trace_committed_meta(
            sample_ids=meta.sample_ids,
            tags=meta.tags,
            extra_info=meta.extra_info,
            plan=plan,
            pending_identity=pending["pending_identity"],
            execution_ids_by_rollout=pending["controller_summary"][
                "execution_ids_by_rollout"
            ],
        )
        fields = [
            "input_ids",
            "input_lengths",
            "generation_logprobs",
            "token_mask",
            "sample_mask",
            "advantages",
        ]
        if ROUTED_EXPERTS_FIELD in preparation["materialization"]["train_data"]:
            fields.append(ROUTED_EXPERTS_FIELD)
        if not skip_policy_logprobs:
            fields.append("prev_logprobs")
        if not skip_reference_logprobs:
            fields.append("reference_policy_logprobs")
        observed = read_columns(
            self._dp_client,
            meta,
            select_fields=fields,
            pad_value_dict={"input_ids": self.tokenizer.pad_token_id},
        )
        expected = preparation["materialization"]["train_data"]
        expected_width = int(expected["input_ids"].shape[1])
        aligned: dict[str, torch.Tensor] = {}
        for field in fields:
            value = observed[field]
            expected_value = expected.get(field)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"TQ exact-trace field {field!r} is not a tensor")
            if value.ndim >= 2 and value.shape[1] > expected_width:
                trailing = value[:, expected_width:]
                pad_value = self.tokenizer.pad_token_id if field == "input_ids" else 0
                if torch.any(trailing != pad_value):
                    raise ValueError(
                        f"TQ exact-trace field {field!r} has non-canonical padding"
                    )
                value = value[:, :expected_width]
            aligned[field] = value
            if expected_value is not None:
                if not isinstance(expected_value, torch.Tensor) or not torch.equal(
                    value,
                    expected_value,
                ):
                    raise ValueError(
                        f"TQ exact-trace field {field!r} changed after commit"
                    )

        scored = attach_precomputed_trace_logprobs(
            preparation,
            policy_output=(
                {"logprobs": aligned["prev_logprobs"]}
                if not skip_policy_logprobs
                else None
            ),
            reference_output=(
                {"reference_logprobs": aligned["reference_policy_logprobs"]}
                if not skip_reference_logprobs
                else None
            ),
            skip_policy_logprobs=skip_policy_logprobs,
            skip_reference_logprobs=skip_reference_logprobs,
        )
        canonical_train_data = scored["train_data"]
        skipped_fields: dict[str, torch.Tensor] = {}
        canonical_sample_mask = canonical_train_data["sample_mask"]
        if not torch.equal(aligned["sample_mask"], canonical_sample_mask):
            skipped_fields["sample_mask"] = canonical_sample_mask
        for skipped, field in (
            (skip_policy_logprobs, "prev_logprobs"),
            (skip_reference_logprobs, "reference_policy_logprobs"),
        ):
            canonical = canonical_train_data[field]
            if skipped:
                # No worker wrote this field, so seed its canonical zero
                # placeholder exactly once.
                skipped_fields[field] = canonical
            elif not torch.equal(aligned[field], canonical):
                raise ValueError(
                    f"TQ exact-trace worker field {field!r} was not canonical "
                    "on masked token positions"
                )
        if skipped_fields:
            write_columns(
                self._dp_client,
                meta,
                skipped_fields,
            )
        return {
            "plan_id": plan["plan_id"],
            "total_row_count": plan["total_row_count"],
            "logical_rollout_count": plan["logical_rollout_count"],
            "eligible_action_token_count": plan["eligible_action_token_count"],
            "rollout_sequence_mask_metrics": scored.get(
                "rollout_sequence_mask_metrics",
                {},
            ),
        }

    def abort_pending_exact_trace(self, *, pending_handle: str) -> bool:
        """Clear a prepared/committed plan after any controller-side failure."""
        pending = self._exact_trace_pending
        if pending is None:
            failed_identity = self._exact_trace_failed_identity
            if (
                failed_identity is not None
                and failed_identity["pending_handle"] == pending_handle
            ):
                self._exact_trace_failed_identity = None
                return True
            return False
        pending = self._require_exact_trace_pending(pending_handle)
        meta = pending["committed_meta"]
        intended_sample_ids = pending["intended_sample_ids"]
        partition_id = pending["partition_id"]
        if meta is not None:
            self._dp_client.clear_samples(
                sample_ids=meta.sample_ids,
                partition_id=meta.partition_id,
            )
        elif intended_sample_ids is not None and partition_id is not None:
            self._dp_client.clear_samples(
                sample_ids=list(intended_sample_ids),
                partition_id=partition_id,
            )
        # Do not discard the cleanup authority when clear_samples raises.
        # The controller can clear the same deterministic IDs and then retry
        # this idempotent abort to release actor memory.
        self._exact_trace_pending = None
        self._exact_trace_failed_identity = None
        return True

    def finalize_pending_exact_trace(self, *, pending_handle: str) -> None:
        """Release actor memory after the controller cleared committed samples."""
        pending = self._require_exact_trace_pending(pending_handle)
        if pending["commit_state"] != "committed" or pending["committed_meta"] is None:
            raise RuntimeError("Cannot finalize an uncommitted exact-trace plan")
        self._exact_trace_pending = None
        self._exact_trace_failed_identity = None

    def _require_exact_trace_pending(self, pending_handle: str) -> dict[str, Any]:
        pending = self._exact_trace_pending
        if pending is None:
            raise RuntimeError("SyncRolloutActor has no pending exact-trace plan")
        if pending["pending_identity"]["pending_handle"] != pending_handle:
            raise ValueError("Exact-trace pending handle does not match actor state")
        return pending

    def shutdown(self) -> None:
        try:
            if self._exact_trace_pending is not None:
                self.abort_pending_exact_trace(
                    pending_handle=self._exact_trace_pending["pending_identity"][
                        "pending_handle"
                    ]
                )
            self._exact_trace_failed_identity = None
            self._dp_client.close()
        except Exception:
            pass
