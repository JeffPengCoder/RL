# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_exact_trace.yaml",
        "examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_recommended.yaml",
    ],
)
def test_exact_trajectory_example_config_passes_runtime_model_validation(
    monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    repo_root = Path(__file__).parents[2]
    monkeypatch.setenv("NANO_OMNI_MODEL_NAME", "/models/nano-omni")
    monkeypatch.setenv("OSWORLD_TRAIN_DATA", "/data/train.jsonl")
    monkeypatch.setenv("OSWORLD_VALIDATION_DATA", "/data/validation.jsonl")
    monkeypatch.setenv("CC_SMOKE_DATA_PATH", "/data/context-compaction.jsonl")
    register_omegaconf_resolvers()

    raw_config = load_config(repo_root / relative_path)
    resolved_config = OmegaConf.to_container(raw_config, resolve=True)

    assert isinstance(resolved_config, dict)
    config = MasterConfig(**resolved_config)
    assert all(value is not None for value in config.logger["wandb"].values())
    if "osworld_" in relative_path:
        assert config.policy["router_replay"]["enabled"] is True
        assert config.policy["sequence_packing"]["enabled"] is False
    osworld_agent = config.env["nemo_gym"]["osworld_simple_agent"][
        "responses_api_agents"
    ]["osworld_agent"]
    assert osworld_agent["agent_kwargs"]["max_image_history_length"] == 3
    assert osworld_agent["agent_kwargs"]["max_live_images"] == 10
    assert (
        config.policy["generation"]["vllm_kwargs"]["limit_mm_per_prompt"]["image"] == 10
    )

    if relative_path.endswith("osworld_recommended.yaml"):
        assert config.grpo.num_prompts_per_step == 128
        assert config.grpo.num_generations_per_prompt == 16
        assert config.grpo.max_rollout_turns == 200
        assert config.grpo.use_leave_one_out_baseline is True
        assert config.grpo.async_grpo.enabled is True
        assert config.loss_fn.use_importance_sampling_correction is True
        assert config.loss_fn.truncated_importance_sampling_type is None
        assert config.policy["train_global_batch_size"] == 2048
        assert config.policy["max_total_sequence_length"] == 49152
        assert config.policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
        assert config.policy["megatron_cfg"]["context_parallel_size"] == 8
        assert config.policy["megatron_cfg"]["expert_model_parallel_size"] == 8
        assert config.policy["megatron_cfg"]["optimizer"]["lr"] == pytest.approx(5e-6)
        assert config.policy["generation"]["colocated"]["resources"]["num_nodes"] == 2
        assert config.cluster["num_nodes"] == 2
        assert config.data["validation"] is None
