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

"""Failure-injection tests for the non-idempotent sync exact/TQ boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, **kwargs):
        return self._fn(**kwargs)


def _bare_rollout_actor():
    from nemo_rl.experience.sync_rollout_actor import SyncRolloutActor

    actor_type = SyncRolloutActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_type)
    actor._exact_trace_pending = None
    actor._exact_trace_failed_identity = None
    return actor


def test_run_success_then_logger_failure_never_replays_run(monkeypatch):
    import nemo_rl.algorithms.grpo as grpo_mod
    import nemo_rl.algorithms.utils as algorithm_utils
    import nemo_rl.data.llm_message_utils as message_utils
    import nemo_rl.experience.rollouts as rollouts_mod
    import nemo_rl.experience.sync_exact_trace as exact_mod
    import nemo_rl.experience.sync_rollout_actor as actor_mod
    import nemo_rl.experience.trace_batch_scoring as scoring_mod

    actor = _bare_rollout_actor()
    actor.master_config = SimpleNamespace(
        policy={
            "generation": {},
            "make_sequence_length_divisible_by": 1,
            "router_replay": {"enabled": False},
        },
        logger={"wandb_enabled": False, "wandb": {}},
        env={"nemo_gym": {}},
        reward_penalties={},
        grpo=SimpleNamespace(use_leave_one_out_baseline=False),
    )
    actor.tokenizer = SimpleNamespace(pad_token_id=0)
    actor.task_to_env = {}

    class _Generation:
        def clear_logger_metrics(self):
            pass

        def finish_generation(self):
            pass

        def get_logger_metrics(self):
            raise RuntimeError("logger collection failed")

    actor.policy_generation = _Generation()
    logical_logs = [
        [
            {"role": "user", "content": "p", "token_ids": torch.tensor([1])},
            {
                "role": "assistant",
                "content": "a",
                "token_ids": torch.tensor([2]),
                "generation_logprobs": torch.tensor([-0.1]),
            },
        ],
        [
            {"role": "user", "content": "p", "token_ids": torch.tensor([1])},
            {
                "role": "assistant",
                "content": "b",
                "token_ids": torch.tensor([3]),
                "generation_logprobs": torch.tensor([-0.2]),
            },
        ],
    ]
    final_batch = BatchedDataDict(
        {
            "message_log": logical_logs,
            "physical_message_logs": [[row] for row in logical_logs],
            "rollout_trace_bundle": [{"rollout_id": "r0"}, {"rollout_id": "r1"}],
            "rollout_execution_context": [
                {"execution_id": "e0"},
                {"execution_id": "e1"},
            ],
            "length": torch.tensor([1, 1]),
            "total_reward": torch.tensor([0.0, 1.0]),
        }
    )
    run_calls = 0

    def _run(**kwargs):
        nonlocal run_calls
        del kwargs
        run_calls += 1
        return SimpleNamespace(final_batch=final_batch, rollout_metrics={})

    plan = {
        "plan_id": "plan",
        "rollout_ids": ["r0", "r1"],
        "total_row_count": 2,
        "physical_trace_count": 2,
        "padding_row_count": 0,
        "logical_rollout_count": 2,
        "eligible_action_token_count": 2,
    }
    train_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [1, 3]]),
            "input_lengths": torch.tensor([2, 2]),
            "generation_logprobs": torch.tensor([[0.0, -0.1], [0.0, -0.2]]),
            "token_mask": torch.tensor([[0, 1], [0, 1]]),
            "sample_mask": torch.ones(2),
            "advantages": torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
        }
    )
    preparation = {
        "plan": plan,
        "materialization": {
            "train_data": train_data,
            "materialized_message_logs": logical_logs,
            "row_rewards": torch.tensor([0.0, 1.0]),
        },
        "rollout_advantages": {"r0": -1.0, "r1": 1.0},
    }
    summary = {
        "pending_identity": {
            "schema_version": 1,
            "pending_handle": "unused-by-first-call",
            "sampling_event_id": "event",
            "generation_policy_version": "policy",
            "optimizer_step_id": "step",
            "logical_rollout_count": 2,
            "group_size": 2,
        },
        "plan_id": "plan",
        "training_admission_contract_id": "admission",
        "total_row_count": 2,
        "physical_trace_count": 2,
        "padding_row_count": 0,
        "logical_rollout_count": 2,
        "eligible_action_token_count": 2,
        "scheduler_step_increment": 2,
        "execution_ids_by_rollout": {"r0": "e0", "r1": "e1"},
    }

    monkeypatch.setattr(grpo_mod, "_should_use_nemo_gym", lambda cfg: True)
    monkeypatch.setattr(grpo_mod, "_create_advantage_estimator", lambda cfg: object())
    monkeypatch.setattr(
        grpo_mod,
        "extract_initial_prompt_messages",
        lambda message_log, lengths: [row[:1] for row in message_log],
    )
    monkeypatch.setattr(
        message_utils,
        "batched_message_log_to_flat_message",
        lambda *args, **kwargs: (
            BatchedDataDict({"token_ids": torch.tensor([[1], [1]])}),
            torch.tensor([1, 1]),
        ),
    )
    monkeypatch.setattr(rollouts_mod, "backfill_missing_routed_experts", lambda x: None)
    monkeypatch.setattr(actor_mod, "run_nemo_gym_rollout_sync", _run)
    monkeypatch.setattr(
        scoring_mod,
        "prepare_trace_batch_for_scoring",
        lambda *args, **kwargs: preparation,
    )
    monkeypatch.setattr(
        exact_mod,
        "summarize_exact_trace_plan",
        lambda *args, **kwargs: summary,
    )
    monkeypatch.setattr(
        algorithm_utils,
        "calculate_baseline_and_std_per_prompt",
        lambda *args, **kwargs: (torch.tensor([0.5, 0.5]), torch.ones(2)),
    )

    input_batch = BatchedDataDict({"message_log": logical_logs})
    kwargs = {
        "generation_policy_version": "policy",
        "sampling_event_id": "event",
        "group_size": 2,
        "batch_quantum": 2,
        "optimizer_step_id": "step",
        "first_iter": False,
    }
    with pytest.raises(RuntimeError, match="logger collection failed"):
        actor.prepare_pending_exact_trace(input_batch, **kwargs)
    with pytest.raises(RuntimeError, match="refusing an implicit replay"):
        actor.prepare_pending_exact_trace(input_batch, **kwargs)

    assert run_calls == 1
    assert actor._exact_trace_pending is None
    assert actor._exact_trace_failed_identity is not None


def test_lost_first_put_ack_is_not_replayed_and_plan_ids_are_cleared(monkeypatch):
    import nemo_rl.experience.sync_exact_trace as exact_mod
    import nemo_rl.experience.sync_rollout_actor as actor_mod

    actor = _bare_rollout_actor()
    actor.master_config = SimpleNamespace(
        policy={"make_sequence_length_divisible_by": 1}
    )
    clear_calls = []
    actor._dp_client = SimpleNamespace(
        clear_samples=lambda **kwargs: clear_calls.append(kwargs)
    )
    pending_identity = {
        "pending_handle": "handle",
        "sampling_event_id": "event",
        "generation_policy_version": "policy",
    }
    actor._exact_trace_pending = {
        "pending_identity": pending_identity,
        "controller_summary": {"execution_ids_by_rollout": {"r0": "e0"}},
        "preparation": {
            "plan": {"plan_id": "plan"},
            "materialization": {
                "train_data": BatchedDataDict(
                    {
                        "input_ids": torch.tensor([[1, 2]]),
                        "input_lengths": torch.tensor([2]),
                        "sample_mask": torch.ones(1),
                    }
                )
            },
        },
        "commit_state": "prepared",
        "committed_meta": None,
        "partition_id": None,
        "intended_sample_ids": None,
    }
    monkeypatch.setattr(
        exact_mod,
        "build_exact_trace_wire_identity",
        lambda *args, **kwargs: (["plan:0"], [{"row_index": 0}], {"plan_id": "plan"}),
    )
    put_calls = 0

    def _lost_ack(*args, **kwargs):
        nonlocal put_calls
        del args, kwargs
        put_calls += 1
        raise RuntimeError("put ACK lost")

    monkeypatch.setattr(actor_mod, "kv_first_write", _lost_ack)
    monkeypatch.setattr(actor_mod, "trace_rollout_payload", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="put ACK lost"):
        actor.commit_pending_exact_trace(
            pending_handle="handle",
            partition_id="train",
        )
    with pytest.raises(RuntimeError, match="commit outcome is ambiguous"):
        actor.commit_pending_exact_trace(
            pending_handle="handle",
            partition_id="train",
        )

    assert put_calls == 1
    assert actor.abort_pending_exact_trace(pending_handle="handle")
    assert clear_calls == [{"sample_ids": ["plan:0"], "partition_id": "train"}]


def test_optimizer_rpc_ambiguity_is_never_reported_as_pre_update(monkeypatch):
    import nemo_rl.algorithms.grpo_sync as sync_mod

    plan = {
        "plan_id": "plan",
        "total_row_count": 2,
        "logical_rollout_count": 1,
    }
    cleanup_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["plan:0", "plan:1"],
    )
    step_record = {"optimizer_step_id": "step", "plan_id": "plan"}
    monkeypatch.setattr(
        sync_mod,
        "_build_exact_trace_controller_authority",
        lambda **kwargs: (plan, cleanup_meta, step_record),
    )
    monkeypatch.setattr(sync_mod.ray, "get", lambda value: value)
    monkeypatch.setattr(
        sync_mod,
        "_resolve_logprob_skip_flags",
        lambda config: (True, True),
    )
    monkeypatch.setattr(
        sync_mod,
        "_should_log_nemo_gym_responses",
        lambda config: True,
    )

    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["plan:0", "plan:1"],
        sequence_lengths=[2, 2],
    )
    abort_calls = []
    actor = SimpleNamespace(
        commit_pending_exact_trace=_RemoteMethod(lambda **kwargs: meta),
        validate_pending_exact_trace_scoring=_RemoteMethod(
            lambda **kwargs: {"plan_id": "plan"}
        ),
        abort_pending_exact_trace=_RemoteMethod(
            lambda **kwargs: abort_calls.append(kwargs) or True
        ),
        finalize_pending_exact_trace=_RemoteMethod(lambda **kwargs: None),
    )

    class _Policy:
        tq_partition_id = "train"

        def __init__(self):
            self.train_calls = 0

        def prepare_step(self, **kwargs):
            pass

        def read_from_dataplane(self, meta, select_fields, pad_value_dict):
            del meta, select_fields, pad_value_dict
            return BatchedDataDict(
                {
                    "input_lengths": torch.tensor([2, 2]),
                    "generation_logprobs": torch.zeros(2, 2),
                    "token_mask": torch.tensor([[0, 1], [0, 0]]),
                    "sample_mask": torch.tensor([1.0, 0.0]),
                    "advantages": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
                    "prev_logprobs": torch.zeros(2, 2),
                    "reference_policy_logprobs": torch.zeros(2, 2),
                }
            )

        def prepare_for_training(self):
            pass

        def train_from_meta(self, *args, **kwargs):
            del args, kwargs
            self.train_calls += 1
            raise RuntimeError("worker updated, response lost")

        def finish_step(self, meta):
            del meta

    policy = _Policy()

    class _NoopContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    timer = SimpleNamespace(time=lambda name: _NoopContext())
    master_config = SimpleNamespace(
        policy={"train_micro_batch_size": 1},
    )
    controller_summary = {
        "pending_identity": {"pending_handle": "handle"},
        "row_rewards": [1.0, 0.0],
    }

    with pytest.raises(RuntimeError, match="outcome is ambiguous") as error:
        sync_mod._execute_pending_exact_trace_step(
            policy=policy,
            rollout_actor=actor,
            controller_summary=controller_summary,
            loss_fn=object(),
            master_config=master_config,
            timer=timer,
            pad_value_dict={"input_ids": 0},
            sync_kv_scales=False,
        )

    assert "do not replay this optimizer_step_id" in str(error.value)
    assert policy.train_calls == 1
    assert abort_calls == [{"pending_handle": "handle"}]
