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

"""Cross-repository tests for the current Gym OSWorld exact-trace contract.

The producer side deliberately uses the Gym implementation pinned by this
NeMo-RL checkout.  The consumer side uses the real NeMo-RL postprocessor and
serialized trace validator.  No historical scripted-agent fixture is involved,
so a passing test proves that the source graph named by the current gitlink is
internally compatible.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.environments.nemo_gym_trace import validate_rollout_trace_bundle
from responses_api_agents.osworld_agent.app import (
    OSWorldRunRequest,
    _build_response,
)


_IMAGE_A = "data:image/png;base64,QQ=="
_IMAGE_B = "data:image/png;base64,Qg=="
_IMAGE_C = "data:image/png;base64,Qw=="
_IMAGE_D = "data:image/png;base64,RA=="


class _Tokenizer:
    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        return [" ".join(map(str, token_ids)) for token_ids in batch]


class _MockNemoGymActor:
    cfg: dict[str, Any] = {}


def _identity(rollout_index: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rollout_id": f"bridge-rollout-{rollout_index}",
        "group_id": "bridge-group",
        "task_id": f"bridge-task-{rollout_index}",
        "rollout_index": rollout_index,
        "attempt_index": 0,
    }


def _token_and_media_views(
    *, rewrite: bool, ordered_pair: tuple[str, str]
) -> tuple[list[list[int]], list[list[str]]]:
    if not rewrite:
        prompts: list[list[int]] = []
        context: list[int] = []
        for turn_id in range(1, 6):
            prompt = [*context, 1000 + turn_id]
            prompts.append(prompt)
            context = [*prompt, 2000 + turn_id]
        return prompts, [[_IMAGE_A] for _ in range(5)]

    # Calls 1-2 and 3-4 are prefix-contiguous. Calls 3 and 5 deliberately
    # replace both the token prefix and the image view, forcing new traces.
    return (
        [
            [101],
            [101, 201, 102],
            [301],
            [301, 401, 302],
            [501],
        ],
        [
            [_IMAGE_A],
            [_IMAGE_A],
            [*ordered_pair],
            [*ordered_pair],
            [_IMAGE_D],
        ],
    )


def _steps(
    *, rewrite: bool, ordered_pair: tuple[str, str] = (_IMAGE_B, _IMAGE_C)
) -> list[dict[str, Any]]:
    prompt_views, media_views = _token_and_media_views(
        rewrite=rewrite,
        ordered_pair=ordered_pair,
    )
    generation_ids = (
        [201, 202, 401, 402, 601] if rewrite else [2001, 2002, 2003, 2004, 2005]
    )
    steps = []
    for step_index, (prompt_ids, image_urls, generation_id) in enumerate(
        zip(prompt_views, media_views, generation_ids)
    ):
        turn_id = step_index + 1
        prompt_messages = [
            {"role": "system", "content": "Operate the desktop."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"observation {turn_id}"},
                    *[
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "high",
                        }
                        for image_url in image_urls
                    ],
                ],
            },
        ]
        raw_completion = f"assistant turn {turn_id}"
        parsed_actions = [{"type": "computer_initialize_state"}]
        steps.append(
            {
                "step": step_index,
                "state": {"screenshot_sha256": f"screen-{turn_id}"},
                "next_state": {"screenshot_sha256": f"screen-{turn_id + 1}"},
                "model_text": raw_completion,
                "actions": parsed_actions,
                "reward": float(turn_id == 5),
                "done": turn_id == 5,
                "info": {
                    "agent": {
                        "model_calls": [
                            {
                                "parse_attempt": 1,
                                "prompt_messages": prompt_messages,
                                "response": {
                                    "prompt_token_ids": prompt_ids,
                                    "generation_token_ids": [generation_id],
                                    "generation_log_probs": [-0.1 * turn_id],
                                    "finish_reason": "stop",
                                    "raw_content": raw_completion,
                                },
                                "accepted": True,
                                "parse_error": None,
                                "parsed_actions": parsed_actions,
                            }
                        ]
                    }
                },
            }
        )
    return steps


def _serialized_gym_result(
    *,
    rewrite: bool,
    rollout_index: int = 0,
    ordered_pair: tuple[str, str] = (_IMAGE_B, _IMAGE_C),
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _identity(rollout_index)
    body = OSWorldRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input="initial text"
        ),
        verifier_metadata={"task_id": identity["task_id"]},
        trajectory_identity=identity,
    )
    response = _build_response(
        body,
        {
            "steps": _steps(rewrite=rewrite, ordered_pair=ordered_pair),
            "score": 1.0,
            "reward": 1.0,
            "finished": True,
            "error": None,
            "termination_reason": "done",
        },
        "dummy-model",
        1.0,
        1.0,
        max_trajectory_length=3,
        max_output_tokens=32,
    )
    row = {
        "_rowidx": rollout_index,
        "trajectory_identity": deepcopy(identity),
    }
    return row, json.loads(response.model_dump_json())


def _postprocess(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    return postprocess(
        _MockNemoGymActor(),
        row,
        result,
        _Tokenizer(),
        generation_only=True,
    )


@pytest.mark.parametrize(
    ("rewrite", "expected_turns"),
    [
        (False, [[1, 2, 3, 4, 5]]),
        (True, [[1, 2], [3, 4], [5]]),
    ],
)
def test_current_gym_json_crosses_current_nemo_rl_postprocessor(
    rewrite: bool,
    expected_turns: list[list[int]],
) -> None:
    row, gym_result = _serialized_gym_result(rewrite=rewrite)
    normalized = _postprocess(row, gym_result)
    bundle = normalized["rollout_trace_bundle"]

    assert bundle["schema_version"] == 3
    assert [
        trace["source_turn_ids"] for trace in bundle["physical_traces"]
    ] == expected_turns
    assert len(normalized["physical_message_logs"]) == len(expected_turns)
    checks = validate_rollout_trace_bundle(
        json.loads(json.dumps(bundle)),
        media_assets=json.loads(json.dumps(gym_result["response"]["media_assets"])),
        strict=True,
    )
    assert checks["model_call_count"] == 5
    assert checks["physical_trace_count"] == len(expected_turns)
    assert checks["sampled_token_count"] == 5
    assert checks["eligible_trainable_token_count"] == 5

    # Trainer transport keeps the semantic trajectory but drops bulky exact
    # evidence and inline media from the compact Ray result.
    full_result = normalized["full_result"]
    assert full_result["nemo_rl_trace_bundle"] == bundle
    assert set(full_result["response"]).isdisjoint(
        {
            "media_assets",
            "completion_evidence",
            "final_policy_decision",
            "lineage_deltas",
        }
    )
    assert not any(
        value.startswith("data:image") for value in _walk_strings(full_result)
    )


def test_current_bridge_rejects_output_evidence_mismatch() -> None:
    row, gym_result = _serialized_gym_result(rewrite=True)
    first_output = next(
        item
        for item in gym_result["response"]["output"]
        if "generation_token_ids" in item
    )
    first_output["generation_token_ids"] = [999999]

    with pytest.raises(
        ValueError,
        match="does not exactly match the generation response",
    ):
        _postprocess(row, gym_result)


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def test_current_bridge_preserves_same_shape_media_order() -> None:
    ordered_pair = (_IMAGE_C, _IMAGE_B)
    row, gym_result = _serialized_gym_result(
        rewrite=True,
        ordered_pair=ordered_pair,
    )
    media_assets = gym_result["response"]["media_assets"]
    turn_three = _postprocess(row, gym_result)["rollout_trace_bundle"]["model_calls"][2]
    observed_urls = [
        media_assets[media_id]["source_part"]["image_url"]
        for media_id in turn_three["media_ids"]
    ]

    assert observed_urls == list(ordered_pair)


def test_current_bridge_keeps_two_rollouts_isolated() -> None:
    normalized_results = []
    for rollout_index in (0, 1):
        row, gym_result = _serialized_gym_result(
            rewrite=True,
            rollout_index=rollout_index,
        )
        normalized_results.append(_postprocess(row, gym_result))

    bundles = [result["rollout_trace_bundle"] for result in normalized_results]
    assert bundles[0]["rollout_id"] != bundles[1]["rollout_id"]
    assert bundles[0]["source_row_index"] == 0
    assert bundles[1]["source_row_index"] == 1
    assert [
        [trace["source_turn_ids"] for trace in bundle["physical_traces"]]
        for bundle in bundles
    ] == [
        [[1, 2], [3, 4], [5]],
        [[1, 2], [3, 4], [5]],
    ]
