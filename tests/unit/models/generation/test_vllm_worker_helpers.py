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

"""Tests for vLLM worker helper functions."""

import pytest

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    resolve_http_request_sampling_contract,
)
from nemo_rl.models.generation.vllm.worker_utils import (
    resolve_data_parallel_local_rank,
    resolve_distributed_executor_backend,
)


def test_http_sampling_contract_keeps_training_on_policy() -> None:
    config = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 768,
        "vllm_cfg": {
            "http_server_evaluation_sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 4096,
            }
        },
    }

    assert resolve_http_request_sampling_contract(config, "training") == {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 768,
    }


def test_http_sampling_contract_allows_only_pinned_evaluation_profile() -> None:
    config = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 768,
        "vllm_cfg": {
            "http_server_evaluation_sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 4096,
            }
        },
    }

    assert resolve_http_request_sampling_contract(config, "evaluation") == {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 4096,
    }


@pytest.mark.parametrize(
    ("tp", "pp", "ep", "expected"),
    [
        (2, 1, 2, "ray"),
        (1, 2, 2, "ray"),
        (1, 1, 8, "uni"),
        (1, 1, 1, None),
    ],
)
def test_resolve_distributed_executor_backend(tp, pp, ep, expected):
    assert resolve_distributed_executor_backend(tp, pp, ep) == expected


@pytest.mark.parametrize(
    ("rank", "model_parallel_size", "executor_backend", "expected"),
    [
        (7, 1, "uni", 0),
        (6, 2, "ray", 3),
    ],
)
def test_resolve_data_parallel_local_rank(
    rank, model_parallel_size, executor_backend, expected
):
    assert (
        resolve_data_parallel_local_rank(rank, model_parallel_size, executor_backend)
        == expected
    )
