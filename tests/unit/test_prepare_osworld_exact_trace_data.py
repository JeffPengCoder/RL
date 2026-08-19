# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from examples.nemo_gym.prepare_osworld_exact_trace_data import (
    annotate_trajectory_rows,
    split_osworld_rows,
)


def _row(task_id: str, domain: str = "chrome") -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": f"task {task_id}"}]
        },
        "verifier_metadata": {
            "task_id": task_id,
            "domain": domain,
            "osworld_task": {"id": task_id, "snapshot": domain},
        },
    }


def test_split_is_fixed_disjoint_and_seeded_by_task_id():
    rows = [_row(f"task-{index}") for index in range(6)]
    train_a, validation_a = split_osworld_rows(
        rows, validation_count=2, seed="fixed-seed"
    )
    train_b, validation_b = split_osworld_rows(
        list(reversed(rows)), validation_count=2, seed="fixed-seed"
    )

    validation_ids_a = {row["verifier_metadata"]["task_id"] for row in validation_a}
    validation_ids_b = {row["verifier_metadata"]["task_id"] for row in validation_b}
    train_ids = {row["verifier_metadata"]["task_id"] for row in train_a}
    assert validation_ids_a == validation_ids_b
    assert validation_ids_a.isdisjoint(train_ids)
    assert len(train_a) == len(train_b) == 4


def test_split_can_keep_the_complete_source_as_training_data():
    rows = [_row(f"task-{index}") for index in range(361)]

    train, validation = split_osworld_rows(
        rows, validation_count=0, seed="unused-for-all-train"
    )

    assert train == rows
    assert validation == []


def test_annotate_adds_exact_trace_identity_without_mutating_source():
    source = _row("task-1")
    [annotated] = annotate_trajectory_rows(
        [source], group_prefix="experiment", agent_name="osworld_simple_agent"
    )

    assert "trajectory_identity" not in source
    assert annotated["trajectory_identity"] == {
        "schema_version": 1,
        "group_id": "experiment:chrome:task-1",
        "task_id": "task-1",
        "rollout_index": 0,
        "attempt_index": 0,
    }
    assert annotated["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "osworld_simple_agent",
    }
