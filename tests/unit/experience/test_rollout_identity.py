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

from copy import deepcopy

import pytest

from nemo_rl.experience.rollout_identity import (
    new_sampling_event_id,
    scope_trajectory_identities,
)


def _row(*, rollout_index: int) -> dict:
    return {
        "trajectory_identity": {
            "schema_version": 1,
            "group_id": "dataset-group-7",
            "task_id": "task-7",
            "rollout_index": rollout_index,
            "attempt_index": 0,
        }
    }


def test_sampling_event_ids_are_unique_even_for_same_step() -> None:
    first = new_sampling_event_id(purpose="validation", step=12)
    second = new_sampling_event_id(purpose="validation", step=12)

    assert first != second
    assert first.startswith("sampling-validation-step-00000012-")


def test_event_scoping_is_retry_idempotent_and_replica_distinct() -> None:
    rows = [_row(rollout_index=0), _row(rollout_index=1)]
    sampling_event_id = new_sampling_event_id(purpose="training")

    scope_trajectory_identities(rows, sampling_event_id=sampling_event_id)
    first_scope = deepcopy(rows)
    scope_trajectory_identities(rows, sampling_event_id=sampling_event_id)

    assert rows == first_scope
    identities = [row["trajectory_identity"] for row in rows]
    assert {identity["sampling_event_id"] for identity in identities} == {
        sampling_event_id
    }
    assert {identity["source_group_id"] for identity in identities} == {
        "dataset-group-7"
    }
    assert len({identity["group_id"] for identity in identities}) == 1
    assert len({identity["rollout_id"] for identity in identities}) == 2


def test_new_event_changes_group_and_rollout_ids() -> None:
    first = [_row(rollout_index=0)]
    second = deepcopy(first)

    scope_trajectory_identities(
        first,
        sampling_event_id=new_sampling_event_id(purpose="training"),
    )
    scope_trajectory_identities(
        second,
        sampling_event_id=new_sampling_event_id(purpose="training"),
    )

    first_identity = first[0]["trajectory_identity"]
    second_identity = second[0]["trajectory_identity"]
    assert first_identity["group_id"] != second_identity["group_id"]
    assert first_identity["rollout_id"] != second_identity["rollout_id"]


def test_scoped_row_rejects_a_different_event() -> None:
    rows = [_row(rollout_index=0)]
    scope_trajectory_identities(rows, sampling_event_id="sampling-training-first")

    with pytest.raises(ValueError, match="different sampling event"):
        scope_trajectory_identities(rows, sampling_event_id="sampling-training-second")


def test_legacy_v2_input_is_normalized_and_event_scoped() -> None:
    source = {
        "context_compaction_contract_version": 2,
        "context_compaction_rollout_id": "rollout-static-from-old-controller",
        "context_compaction_group_id": "dataset-group-7",
        "context_compaction_task_id": "task-7",
        "context_compaction_rollout_index": 3,
        "context_compaction_attempt_index": 0,
    }
    first = deepcopy(source)
    second = deepcopy(source)
    first_rows = [first]
    second_rows = [second]

    scope_trajectory_identities(first_rows, sampling_event_id="sampling-first")
    scope_trajectory_identities(second_rows, sampling_event_id="sampling-second")

    first_identity = first_rows[0]["trajectory_identity"]
    second_identity = second_rows[0]["trajectory_identity"]
    assert first_identity["source_group_id"] == "dataset-group-7"
    assert first_identity["rollout_index"] == 3
    assert first_identity["group_id"] != second_identity["group_id"]
    assert first_identity["rollout_id"] != second_identity["rollout_id"]
    assert set(first_rows[0]).isdisjoint(
        {
            "context_compaction_contract_version",
            "context_compaction_rollout_id",
            "context_compaction_group_id",
            "context_compaction_task_id",
            "context_compaction_rollout_index",
            "context_compaction_attempt_index",
        }
    )

    # Re-applying the same event is retry-idempotent after normalization.
    snapshot = deepcopy(first_rows)
    scope_trajectory_identities(first_rows, sampling_event_id="sampling-first")
    assert first_rows == snapshot
