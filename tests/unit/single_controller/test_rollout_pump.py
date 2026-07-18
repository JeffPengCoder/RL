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

"""End-to-end test: SC._rollout_pump writes the expected rows to TQ."""

from __future__ import annotations

import ray
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerBundle
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import RolloutManager

# Reuse fixtures from the experience tests; same shape as test_async_rollout_manager.
from tests.unit.experience.test_rollout_manager import (
    single_multi_step_calculator_input_sample,  # noqa: F401
)
from tests.unit.experience.test_rollouts import (
    initial_multi_step_calculator_batch,  # noqa: F401
    multi_step_calculator_environment,  # noqa: F401
    multi_step_setup_vllm_async,  # noqa: F401
    rollout_cluster,  # noqa: F401
    rollout_tokenizer,  # noqa: F401
)
from tests.unit.single_controller._dp_fakes import (
    _BULK_FIELDS,
    _PARTITION_ID,
    _SyncDPAdapter,
    _TQActor,
)


def test_rollout_pump_writes_expected_tq_data(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,  # noqa: F811
    tmp_path,
):
    """SC._rollout_pump writes max_rollout_prompts * num_generations rows to TQ with the expected fields and tags."""
    vllm_generation, tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample

    num_generations = 2
    max_rollout_prompts = 2
    # TQReplayBuffer.add writes ``num_generations`` training rows per prompt.
    expected_samples = max_rollout_prompts * num_generations
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    tq_actor = _TQActor.remote(
        partition_id=_PARTITION_ID,
        fields=_BULK_FIELDS,
        num_samples=expected_samples * 4,
        consumer_tasks=["train"],
    )
    dp_adapter = _SyncDPAdapter(tq_actor)

    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": expected_samples},
        grpo={
            "num_prompts_per_step": max_rollout_prompts,
            "num_generations_per_prompt": num_generations,
            "max_num_steps": 1,
            "max_num_epochs": 1,
        },
        async_rl=AsyncRLConfig(
            batch_selection_strategy="staleness_window",
            max_weight_staleness_versions=1,
            min_groups_for_streaming_train=1,
            max_inflight_prompts=max_rollout_prompts,
            max_buffered_rollouts=max_rollout_prompts,
            over_sampling=True,
            force_in_order=False,
        ),
        logger={
            "log_dir": str(tmp_path / "logs"),
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "monitor_gpus": False,
        },
    )
    # Wrap each value in a single-element list so size==1 and v[0] returns the original field.
    batched_sample = BatchedDataDict({k: [v] for k, v in input_sample.items()})
    dataloader = [batched_sample] * max_rollout_prompts

    tq_buffer = TQReplayBuffer(
        dp_adapter,
        partition_id=_PARTITION_ID,
        pad_value_dict={"token_ids": int(tokenizer.pad_token_id or 0)},
    )
    rollout_manager = RolloutManager(
        tokenizer=tokenizer,
        env_handles=env_handles,
        num_generations_per_prompt=num_generations,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        policy_generation=vllm_generation,
        use_nemo_gym=False,
        tq_buffer=tq_buffer,
    )
    bundle = SingleControllerBundle(
        gen_handle=vllm_generation,
        trainer_handle=object(),
        env_handles={},
        train_cluster=None,  # type: ignore[arg-type]  # unused by _rollout_pump
        inference_cluster=None,  # type: ignore[arg-type]
        dp_client=dp_adapter,
        dataloader=dataloader,
        weight_synchronizer=object(),  # type: ignore[arg-type]
        advantage_estimator=None,
        loss_fn=None,  # type: ignore[arg-type]
        rollout_manager=rollout_manager,
        tq_buffer=tq_buffer,
        partition_id=_PARTITION_ID,
    )
    ctrl = SingleControllerActor.remote(master_config=master_config, bundle=bundle)

    vllm_generation.prepare_for_generation()
    ray.get(ctrl._rollout_pump.remote())
    vllm_generation.finish_generation()

    sample_ids = ray.get(tq_actor.list_sample_ids.remote(_PARTITION_ID))
    assert len(sample_ids) == expected_samples

    # pack_payload stamps sample_ids as ``{group_uuid}_g{i}``.
    group_ids: set[str] = set()
    for sid in sample_ids:
        head, _, tail = sid.rpartition("_g")
        assert head and tail.isdigit(), f"unexpected sample_id: {sid}"
        group_ids.add(head)
    assert len(group_ids) == max_rollout_prompts

    bulk = ray.get(
        tq_actor.get_samples.remote(
            sample_ids=sample_ids,
            partition_id=_PARTITION_ID,
            select_fields=_BULK_FIELDS,
        )
    )
    assert set(bulk.keys()) >= set(_BULK_FIELDS), (
        f"missing bulk fields: {set(_BULK_FIELDS) - set(bulk.keys())}"
    )

    input_lengths = bulk["input_lengths"].long()
    assert input_lengths.shape[0] == expected_samples
    assert torch.all(input_lengths > 0)
    assert torch.allclose(
        bulk["sample_mask"].float(),
        torch.ones(expected_samples, dtype=torch.float32),
    )

    # Same deterministic prompt as test_async_rollout_manager: the model
    # solves the calculator task every time -> reward == 1.0 and decoded
    # tail contains " 16".
    rewards = bulk["total_reward"].float().flatten()
    assert rewards.shape == (expected_samples,)
    assert torch.allclose(rewards, torch.ones(expected_samples)), (
        f"expected all rewards == 1.0, got {rewards.tolist()}"
    )

    input_ids = bulk["input_ids"]
    token_mask = bulk["token_mask"]
    for i in range(expected_samples):
        length = int(input_lengths[i])
        decoded = tokenizer.decode(
            input_ids[i, :length].tolist(), skip_special_tokens=False
        )
        assert " 16" in decoded[-64:], (
            f"sample {i}: decoded tail {decoded[-64:]!r} missing ' 16'"
        )
        assert int(token_mask[i, :length].sum().item()) > 0, (
            f"sample {i}: token_mask has no assistant tokens"
        )

    tags = ray.get(
        tq_actor.get_tags.remote(partition_id=_PARTITION_ID, sample_ids=sample_ids)
    )
    for tag in tags:
        assert tag["weight_version"] == 0
        # Slim tag schema: weight_version is the only field producers stamp.
        assert set(tag) == {"weight_version"}
