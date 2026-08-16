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

"""Real TransferQueue round-trip for PackedTensor synthetic columns.

This module is intentionally separate from the dependency-light codec tests:
``importorskip`` must not skip the NoOp coverage when TransferQueue is absent.
The shared fixture runs this test once with the simple backend and once with
``mooncake_cpu``.  The latter is the release-relevant nested-tensor wire gate.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("ray")
pytest.importorskip("transfer_queue")

from nemo_rl.data.multimodal_utils import PackedTensor  # noqa: E402
from nemo_rl.data_plane.column_io import kv_first_write, read_columns  # noqa: E402
from nemo_rl.data_plane.packed_tensor_wire import (  # noqa: E402
    PACKED_TENSOR_WIRE_SCHEMA_KEY,
    describe_packed_tensor_wire,
    packed_tensor_schema_from_extra_info,
    packed_tensor_wire_field_names,
)
from nemo_rl.data_plane.preshard import shard_meta_for_dp  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from tests.unit.experience.test_trace_batch_materialization import (  # noqa: E402
    _fixture,
    _materialize,
    _message_logs,
)


def _assert_packed_equal(actual: PackedTensor, expected: PackedTensor) -> None:
    assert actual.dim_to_pack == expected.dim_to_pack
    assert actual.pad_to_max_shape == expected.pad_to_max_shape
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(
        actual.tensors,
        expected.tensors,
        strict=True,
    ):
        if expected_row is None:
            assert actual_row is None
        else:
            assert actual_row is not None
            assert actual_row.dtype == expected_row.dtype
            assert torch.equal(actual_row, expected_row)


def test_packed_tensor_round_trip_real_tq_backends(tq_client_backends) -> None:
    client = tq_client_backends
    sample_ids = ["packed-tq-a", "packed-tq-b", "packed-tq-c"]
    pixel_values = PackedTensor(
        [
            torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 2, 2),
            None,
            torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 2, 2),
        ],
        dim_to_pack=0,
        pad_to_max_shape=True,
    )
    imgs_sizes = PackedTensor(
        [
            torch.tensor([[2, 2]], dtype=torch.long),
            None,
            torch.tensor([[2, 2], [2, 2]], dtype=torch.long),
        ],
        dim_to_pack=0,
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]],
                dtype=torch.long,
            ),
            "input_lengths": torch.tensor([3, 2, 4], dtype=torch.long),
            "sample_mask": torch.ones(3, dtype=torch.float32),
            "pixel_values": pixel_values,
            "imgs_sizes": imgs_sizes,
        }
    )
    schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None
    fields = [
        "input_ids",
        "input_lengths",
        "sample_mask",
        *packed_tensor_wire_field_names(schema),
    ]
    client.register_partition(
        partition_id="packed-media-backend",
        fields=fields,
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )

    try:
        meta = kv_first_write(
            batch,
            sample_ids=sample_ids,
            dp_client=client,
            partition_id="packed-media-backend",
            extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: schema},
        )
        result = read_columns(client, meta, select_fields=list(meta.fields or []))
        _assert_packed_equal(result["pixel_values"], pixel_values)
        _assert_packed_equal(result["imgs_sizes"], imgs_sizes)

        shards, _ = shard_meta_for_dp(meta, dp_world=3, batch_size=3)
        absent_shard = next(
            shard for shard in shards if shard.sample_ids == ["packed-tq-b"]
        )
        absent_schema = packed_tensor_schema_from_extra_info(
            absent_shard.extra_info
        )
        assert absent_schema is not None
        assert absent_schema["sample_ids"] == ["packed-tq-b"]
        absent_result = read_columns(
            client,
            absent_shard,
            select_fields=list(meta.fields or []),
        )
        assert absent_result["pixel_values"].tensors == [None]
        assert absent_result["imgs_sizes"].tensors == [None]
    finally:
        client.clear_samples(
            sample_ids=sample_ids,
            partition_id="packed-media-backend",
        )


def test_ordinary_tq_media_schema_extension_preserves_first_put(
    tq_client_backends,
) -> None:
    """Ordinary TQ learns processor media keys only after rollout returns."""
    client = tq_client_backends
    sample_ids = ["ordinary-packed-a", "ordinary-packed-b"]
    pixel_values = PackedTensor(
        [
            torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
            torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
        ],
        dim_to_pack=0,
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            "input_lengths": torch.tensor([2, 2], dtype=torch.long),
            "sample_mask": torch.ones(2),
            "pixel_values": pixel_values,
        }
    )
    client.register_partition(
        partition_id="packed-media-ordinary-backend",
        fields=["input_ids", "input_lengths", "sample_mask"],
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )

    try:
        meta = kv_first_write(
            batch,
            sample_ids=sample_ids,
            dp_client=client,
            partition_id="packed-media-ordinary-backend",
        )
        schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
        assert schema is not None
        client.ensure_partition_fields(
            "packed-media-ordinary-backend",
            packed_tensor_wire_field_names(schema),
        )
        result = read_columns(client, meta, select_fields=list(meta.fields or []))
        _assert_packed_equal(result["pixel_values"], pixel_values)
    finally:
        client.clear_samples(
            sample_ids=sample_ids,
            partition_id="packed-media-ordinary-backend",
        )


def test_exact_visual_trace_materialization_round_trip_real_tq(
    tq_client_backends,
) -> None:
    """Legacy exact physical rows and TQ use one identical media payload."""
    client = tq_client_backends
    bundle = _fixture("without_compaction.json")
    logs = {
        bundle["rollout_id"]: _message_logs(
            bundle,
            visual_trace_indices={0},
        )
    }
    plan, materialization = _materialize(
        [bundle],
        batch_quantum=2,
        logs=logs,
    )
    train_data = materialization["train_data"]
    sample_ids = [
        f"{plan['plan_id']}:{row_index}"
        for row_index in range(plan["total_row_count"])
    ]
    schema = describe_packed_tensor_wire(train_data, sample_ids=sample_ids)
    assert schema is not None
    fields = [
        key for key, value in train_data.items() if isinstance(value, torch.Tensor)
    ]
    fields.extend(packed_tensor_wire_field_names(schema))
    client.register_partition(
        partition_id="packed-media-exact-backend",
        fields=fields,
        num_samples=len(sample_ids),
        consumer_tasks=["prev_lp", "ref_lp", "train"],
    )

    try:
        meta = kv_first_write(
            train_data,
            sample_ids=sample_ids,
            dp_client=client,
            partition_id="packed-media-exact-backend",
            extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: schema},
        )
        result = read_columns(client, meta, select_fields=list(meta.fields or []))
        _assert_packed_equal(
            result["pixel_values"],
            train_data["pixel_values"],
        )
        assert result["sample_mask"].tolist() == [1.0, 0.0]
    finally:
        client.clear_samples(
            sample_ids=sample_ids,
            partition_id="packed-media-exact-backend",
        )
