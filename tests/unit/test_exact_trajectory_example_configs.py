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
        "examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_opensandbox_exact_trace.yaml",
        "examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_opensandbox_exact_trace_tp8pp2.yaml",
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
    if "osworld_exact_trace" in relative_path:
        assert config.policy["router_replay"]["enabled"] is True
        assert config.policy["sequence_packing"]["enabled"] is False
        vllm_cfg = config.policy["generation"]["vllm_cfg"]
        assert (
            vllm_cfg["http_server_serving_chat_kwargs"][
                "chat_template_content_format"
            ]
            == "string"
        )
        assert vllm_cfg["http_server_evaluation_sampling"] == {
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 4096,
        }
        agent_cfg = config.env["nemo_gym"]["osworld_simple_agent"][
            "responses_api_agents"
        ]["osworld_agent"]
        assert agent_cfg["agent_kwargs"]["parse_retries"] == 5
        assert agent_cfg["agent_kwargs_by_rollout_purpose"] == {
            "training": {"parse_retries": 1},
            "evaluation": {"parse_retries": 5},
        }
        if "opensandbox" in relative_path:
            assert agent_cfg["sandbox_spec"]["image"] == "busybox:1.36"
            assert (
                agent_cfg["sandbox_spec"]["provider_options"][
                    "skip_health_check"
                ]
                is True
            )
            assert agent_cfg["sandbox_spec"]["provider_options"]["extensions"][
                "poolRef"
            ] == "osworld-kvm"
        if relative_path.endswith("_tp8pp2.yaml"):
            assert config.cluster["num_nodes"] == 3
            assert config.policy["megatron_cfg"]["tensor_model_parallel_size"] == 8
            assert config.policy["megatron_cfg"]["pipeline_model_parallel_size"] == 2
            assert config.policy["megatron_cfg"]["context_parallel_size"] == 1
