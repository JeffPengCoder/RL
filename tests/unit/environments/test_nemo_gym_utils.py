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
"""Pure-Python (vllm-free) unit tests for NeMo-Gym helpers.

These run in the default L0 suite. Keep this module free of heavy imports
(e.g. vllm) so the fast detector tests are not gated behind the nemo_gym extra.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.nemo_gym import (
    _detect_invalid_tool_call_and_malformed_thinking,
    configure_nemo_gym_component_roots,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
)


@pytest.mark.parametrize(
    ("output_item_dict", "expected_invalid_tool_call", "expected_malformed_thinking"),
    [
        (
            {"content": [{"text": "use <tool_call>{}</tool_call>"}]},
            True,
            False,
        ),
        (
            {"content": [{"text": "final answer leaked <think>reasoning</think>"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think>"}]},
            False,
            False,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "<think>a</think><think>b"}]},
            False,
            True,
        ),
        (
            {"type": "reasoning", "summary": [{"text": "bad <function_call>{}"}]},
            True,
            False,
        ),
    ],
)
def test_detect_invalid_tool_call_and_malformed_thinking(
    output_item_dict,
    expected_invalid_tool_call,
    expected_malformed_thinking,
):
    assert _detect_invalid_tool_call_and_malformed_thinking(output_item_dict) == (
        expected_invalid_tool_call,
        expected_malformed_thinking,
    )


def test_get_nemo_gym_venv_dir_returns_env_value(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_VENV_DIR", "/opt/gym_venvs")
    assert get_nemo_gym_venv_dir() == "/opt/gym_venvs"


def test_get_nemo_gym_venv_dir_none_when_unset(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_VENV_DIR", raising=False)
    assert get_nemo_gym_venv_dir() is None


def test_get_nemo_gym_uv_cache_dir_none_outside_container(monkeypatch):
    # Outside a container the caller should omit the arg; uv must not be invoked.
    monkeypatch.delenv("NRL_CONTAINER", raising=False)

    def _fail(*args, **kwargs):
        raise AssertionError("uv should not be invoked outside a container")

    monkeypatch.setattr(nemo_gym_mod.subprocess, "check_output", _fail)
    assert get_nemo_gym_uv_cache_dir() is None


def test_get_nemo_gym_uv_cache_dir_uses_uv_inside_container(monkeypatch):
    monkeypatch.setenv("NRL_CONTAINER", "1")
    monkeypatch.setattr(
        nemo_gym_mod.subprocess,
        "check_output",
        lambda *args, **kwargs: b"  /root/.cache/uv\n",
    )
    assert get_nemo_gym_uv_cache_dir() == "/root/.cache/uv"


def test_configure_nemo_gym_component_roots_pins_paired_checkout(monkeypatch):
    paired_root = (
        Path(nemo_gym_mod.__file__).resolve().parents[2]
        / "3rdparty"
        / "Gym-workspace"
        / "Gym"
    ).resolve()
    monkeypatch.setenv(
        "NEMO_GYM_EXTRA_ROOTS", os.pathsep.join(["/plugin-a", str(paired_root)])
    )
    monkeypatch.delenv("NEMO_GYM_ALLOWED_COMPONENT_ROOTS", raising=False)

    assert configure_nemo_gym_component_roots() == paired_root
    expected = os.pathsep.join([str(paired_root), "/plugin-a"])
    assert os.environ["NEMO_GYM_EXTRA_ROOTS"] == expected
    assert os.environ["NEMO_GYM_ALLOWED_COMPONENT_ROOTS"] == expected
    assert Path(sys.path[0]).resolve() == paired_root
    assert (
        Path(sys.modules["nemo_gym"].__file__)
        .resolve()
        .is_relative_to(paired_root / "nemo_gym")
    )


def test_configure_nemo_gym_component_roots_rejects_preloaded_stale_package(
    monkeypatch,
):
    monkeypatch.setitem(
        sys.modules,
        "nemo_gym",
        SimpleNamespace(
            __file__="/opt/nemo-rl/3rdparty/Gym-workspace/Gym/nemo_gym/__init__.py"
        ),
    )

    with pytest.raises(RuntimeError, match="mixed source stack"):
        configure_nemo_gym_component_roots()
