# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run dependency-light TQ capability checks in the policy actor venv."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest import TestCase, main
from unittest.mock import MagicMock, patch


def _required_root(name: str) -> Path:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"{name} must name the sealed runtime authority")
    return Path(value).resolve(strict=True)


def _module_path(module: ModuleType) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"module {module.__name__} has no file identity")
    return Path(value).resolve(strict=True)


class TestMegatronPolicyWorkerActorRuntime(TestCase):
    """Validate source and dependency provenance before the capability checks."""

    @classmethod
    def setUpClass(cls) -> None:
        import megatron.bridge.training.checkpointing as bridge_checkpointing
        import megatron.core
        import nemo_rl
        import nemo_rl.data_plane as data_plane
        import nemo_rl.data_plane.worker_mixin as worker_mixin
        import nemo_rl.models.policy.workers.megatron_policy_worker as policy_worker
        import ray
        import torch

        source_root = _required_root("B06_SOURCE_ROOT")
        actor_venv = _required_root("B06_EXPECTED_ACTOR_VENV")
        if Path(sys.prefix).resolve(strict=True) != actor_venv:
            raise RuntimeError(
                f"wrong actor venv: prefix={sys.prefix}, expected={actor_venv}"
            )

        source_modules = (
            nemo_rl,
            data_plane,
            worker_mixin,
            policy_worker,
            bridge_checkpointing,
            megatron.core,
        )
        dependency_modules = (torch, ray)
        source_paths = {
            module.__name__: str(_module_path(module)) for module in source_modules
        }
        dependency_paths = {
            module.__name__: str(_module_path(module))
            for module in dependency_modules
        }
        for name, value in source_paths.items():
            if not Path(value).is_relative_to(source_root):
                raise RuntimeError(f"source module escaped mount: {name}={value}")
        for name, value in dependency_paths.items():
            if not Path(value).is_relative_to(actor_venv):
                raise RuntimeError(
                    f"runtime dependency escaped actor venv: {name}={value}"
                )

        print(
            "B06_MEGATRON_ACTOR_PROVENANCE|"
            + json.dumps(
                {
                    "actor_venv": str(actor_venv),
                    "dependency_modules": dependency_paths,
                    "executable": str(Path(sys.executable).resolve(strict=True)),
                    "source_modules": source_paths,
                    "source_root": str(source_root),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    def test_model_cp_slicing_capability_is_detected(self) -> None:
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            _model_slices_context_parallel_inputs,
        )

        class ModelSlicesContextParallelInputs:
            model_slices_context_parallel_inputs = True

        self.assertTrue(
            _model_slices_context_parallel_inputs(
                ModelSlicesContextParallelInputs()
            )
        )
        self.assertFalse(_model_slices_context_parallel_inputs(object()))

    def test_tensorized_transfer_queue_setup_is_idempotent(self) -> None:
        import nemo_rl.data_plane as data_plane
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        worker = object.__new__(MegatronPolicyWorkerImpl)
        worker.model_slices_context_parallel_inputs = True
        worker._dp_client = None
        cfg = object()
        client = object()
        builder = MagicMock(return_value=client)
        with patch.object(data_plane, "build_data_plane_client", builder):
            worker.setup_data_plane(cfg)
            worker.setup_data_plane(cfg)

        self.assertIs(worker._dp_client, client)
        builder.assert_called_once_with(cfg, bootstrap=False)


if __name__ == "__main__":
    main(verbosity=2)
