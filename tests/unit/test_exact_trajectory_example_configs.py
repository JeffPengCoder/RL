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
