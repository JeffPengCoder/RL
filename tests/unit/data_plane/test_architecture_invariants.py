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
"""Minimal behavioral invariants for the data-plane wiring.

* ``examples/run_grpo._select_trainer`` dispatches the legacy trainer
  when ``data_plane`` is absent and the sync trainer when enabled.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_run_grpo_dispatches_both_trainers():
    """Check trainer selection for legacy and TransferQueue paths.

    ``examples/run_grpo._select_trainer`` returns the TQ-mediated
    ``grpo_train_sync`` iff ``data_plane.enabled`` is true, and the
    legacy ``grpo_train`` otherwise.
    """
    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_nemo_gym_entrypoint_dispatches_both_sync_trainers() -> None:
    """The Gym entrypoint must not silently bypass the TQ trainer."""
    sys.path.insert(0, str(REPO / "examples" / "nemo_gym"))
    try:
        from run_grpo_nemo_gym import _select_sync_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_sync_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_sync_trainer(cfg_sync) is grpo_train_sync


def test_nemo_gym_entrypoint_builds_tq_policy_factory() -> None:
    """The TQ trainer and TQ policy must be selected as one contract."""
    sys.path.insert(0, str(REPO / "examples" / "nemo_gym"))
    try:
        from run_grpo_nemo_gym import _select_policy_factory
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_policy_factory(cfg_legacy) is None

    dp_cfg = {"enabled": True, "backend": "simple"}
    cfg_sync = MasterConfig.model_construct(data_plane=dp_cfg)
    tq_policy = MagicMock(name="tq_policy")
    with patch(
        "nemo_rl.models.policy.tq_policy.TQPolicy", return_value=tq_policy
    ) as tq_policy_cls:
        factory = _select_policy_factory(cfg_sync)
        assert factory is not None
        assert factory(cluster="cluster", config="config") is tq_policy

    tq_policy_cls.assert_called_once_with(
        cluster="cluster", config="config", dp_cfg=dp_cfg
    )


def test_sync_trainer_scopes_generation_replicas_before_rollout() -> None:
    """TQ dispatch must not send duplicate base rollout indices to Gym."""
    source_path = REPO / "nemo_rl" / "algorithms" / "grpo_sync.py"
    module = ast.parse(source_path.read_text())
    trainer = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "grpo_train_sync"
    )
    calls = [node for node in ast.walk(trainer) if isinstance(node, ast.Call)]
    replica_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_assign_trajectory_generation_replica_indices"
    ]
    rollout_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and "rollout_to_tq" in ast.unparse(node.func)
    ]
    assert len(replica_calls) == 1
    assert len(rollout_calls) == 1
    assert replica_calls[0].lineno < rollout_calls[0].lineno
    generation_only = next(
        keyword.value
        for keyword in rollout_calls[0].keywords
        if keyword.arg == "generation_only"
    )
    assert isinstance(generation_only, ast.Constant)
    assert generation_only.value is False
    generation_policy_version = next(
        keyword.value
        for keyword in rollout_calls[0].keywords
        if keyword.arg == "generation_policy_version"
    )
    assert isinstance(generation_policy_version, ast.Name)
    assert generation_policy_version.id == "generation_policy_version"
    sampling_event_id = next(
        keyword.value
        for keyword in rollout_calls[0].keywords
        if keyword.arg == "sampling_event_id"
    )
    assert isinstance(sampling_event_id, ast.Name)
    assert sampling_event_id.id == "training_sampling_event_id"


def test_sync_rollout_actor_requires_and_forwards_generation_only() -> None:
    """The actor cannot silently fall back to Gym's training default."""
    source_path = REPO / "nemo_rl" / "experience" / "sync_rollout_actor.py"
    module = ast.parse(source_path.read_text())
    actor = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SyncRolloutActor"
    )
    method = next(
        node
        for node in actor.body
        if isinstance(node, ast.FunctionDef) and node.name == "rollout_to_tq"
    )
    keyword_only = {
        argument.arg: default
        for argument, default in zip(
            method.args.kwonlyargs,
            method.args.kw_defaults,
        )
    }
    for required in (
        "generation_only",
        "generation_policy_version",
        "sampling_event_id",
    ):
        assert required in keyword_only
        assert keyword_only[required] is None

    gym_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_nemo_gym_rollout_sync"
    )
    forwarded = next(
        keyword.value
        for keyword in gym_call.keywords
        if keyword.arg == "generation_only"
    )
    assert isinstance(forwarded, ast.Name)
    assert forwarded.id == "generation_only"
    forwarded_policy_version = next(
        keyword.value
        for keyword in gym_call.keywords
        if keyword.arg == "generation_policy_version"
    )
    assert isinstance(forwarded_policy_version, ast.Name)
    assert forwarded_policy_version.id == "generation_policy_version"
    forwarded_sampling_event = next(
        keyword.value
        for keyword in gym_call.keywords
        if keyword.arg == "sampling_event_id"
    )
    assert isinstance(forwarded_sampling_event, ast.Name)
    assert forwarded_sampling_event.id == "sampling_event_id"


def test_sync_trainer_rejects_message_level_advantage_penalties():
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    cfg_disabled = MasterConfig.model_construct(grpo=GRPOConfig())
    _raise_if_message_level_advantage_penalties_enabled(cfg_disabled)

    cfg_enabled = MasterConfig.model_construct(
        grpo=GRPOConfig(
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="grpo.invalid_tool_call_advantage",
    ):
        _raise_if_message_level_advantage_penalties_enabled(cfg_enabled)


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "ensure_partition_fields",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "clear_samples",
        "check_consumption_status",
        "close",
    ],
)
def test_data_plane_client_abc_method_present(method: str) -> None:
    """Keep the ``DataPlaneClient`` ABC swap surface stable.

    A silent rename is a breaking change for every adapter.
    """
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    assert hasattr(DataPlaneClient, method), (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        "This is a breaking change for every adapter."
    )
