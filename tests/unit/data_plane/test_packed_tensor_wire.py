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

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.codec import materialize, pack_jagged_fields
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.packed_tensor_wire import (
    PACKED_TENSOR_WIRE_SCHEMA_KEY,
    concat_packed_tensor_wire_schemas,
    decode_packed_tensor_wire,
    describe_packed_tensor_wire,
    encode_packed_tensor_wire,
    extend_fields_with_packed_tensor_wire,
    packed_tensor_broadcast_components,
    packed_tensor_from_broadcast_components,
    packed_tensor_schema_from_extra_info,
    packed_tensor_wire_field_names,
    subset_packed_tensor_wire_schema,
    validate_packed_tensor_wire_schema,
)
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _media_batch() -> tuple[BatchedDataDict, list[str]]:
    sample_ids = ["sample-a", "sample-b", "sample-c"]
    batch = BatchedDataDict(
        {
            "input_ids": torch.tensor(
                [[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]],
                dtype=torch.long,
            ),
            "input_lengths": torch.tensor([3, 2, 4], dtype=torch.long),
            "sample_mask": torch.ones(3, dtype=torch.float32),
            "pixel_values": PackedTensor(
                [
                    torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
                    None,
                    torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2),
                ],
                dim_to_pack=0,
                pad_to_max_shape=True,
            ),
            "imgs_sizes": PackedTensor(
                [
                    torch.tensor([[2, 2]], dtype=torch.long),
                    None,
                    torch.tensor([[2, 2], [2, 2]], dtype=torch.long),
                ],
                dim_to_pack=0,
            ),
        }
    )
    return batch, sample_ids


def _assert_same_packed(actual: PackedTensor, expected: PackedTensor) -> None:
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


def test_packed_tensor_wire_round_trip_through_noop_data_plane() -> None:
    batch, sample_ids = _media_batch()
    expected_pixel_values = batch["pixel_values"]
    expected_img_sizes = batch["imgs_sizes"]
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["input_ids", "input_lengths", "sample_mask"],
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )

    meta = kv_first_write(
        batch,
        sample_ids=sample_ids,
        dp_client=client,
        partition_id="train",
    )
    schema = packed_tensor_schema_from_extra_info(meta.extra_info)
    assert schema is not None
    assert schema["sample_ids"] == sample_ids
    assert set(packed_tensor_wire_field_names(schema)).issubset(meta.fields or [])
    assert "pixel_values" not in (meta.fields or [])
    assert "imgs_sizes" not in (meta.fields or [])

    result = read_columns(client, meta, select_fields=list(meta.fields or []))
    assert isinstance(result["pixel_values"], PackedTensor)
    assert isinstance(result["imgs_sizes"], PackedTensor)
    _assert_same_packed(result["pixel_values"], expected_pixel_values)
    _assert_same_packed(result["imgs_sizes"], expected_img_sizes)
    assert not any(key.startswith("__nrl_packed_tensor_v1_") for key in result)


def test_packed_tensor_wire_rehashes_dp_subsets_by_sample_id() -> None:
    batch, sample_ids = _media_batch()
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["input_ids", "input_lengths", "sample_mask"],
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )
    meta = kv_first_write(
        batch,
        sample_ids=sample_ids,
        dp_client=client,
        partition_id="train",
    )

    shards, _ = shard_meta_for_dp(meta, dp_world=3, batch_size=3)
    assert sorted(
        sample_id for shard in shards for sample_id in shard.sample_ids
    ) == sorted(sample_ids)
    expected_by_id = {
        sample_id: row
        for sample_id, row in zip(
            sample_ids,
            batch["pixel_values"].tensors,
            strict=True,
        )
    }
    for shard in shards:
        shard_schema = packed_tensor_schema_from_extra_info(shard.extra_info)
        assert shard_schema is not None
        assert shard_schema["sample_ids"] == shard.sample_ids
        result = read_columns(client, shard, select_fields=list(meta.fields or []))
        packed = result["pixel_values"]
        assert isinstance(packed, PackedTensor)
        for sample_id, row in zip(shard.sample_ids, packed.tensors, strict=True):
            expected = expected_by_id[sample_id]
            if expected is None:
                assert row is None
            else:
                assert row is not None and torch.equal(row, expected)


def test_packed_tensor_wire_detects_schema_and_payload_corruption() -> None:
    batch, sample_ids = _media_batch()
    schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None
    corrupted_schema = deepcopy(schema)
    corrupted_schema["entries"][0]["rank"] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_packed_tensor_wire_schema(corrupted_schema)

    wire, encoded_schema = encode_packed_tensor_wire(
        batch,
        sample_ids=sample_ids,
    )
    assert encoded_schema == schema
    values_field = schema["entries"][0]["values_field"]
    corrupted_rows = list(wire[values_field].unbind())
    corrupted_rows[0] = corrupted_rows[0].clone()
    corrupted_rows[0][0] += 1
    wire[values_field] = torch.nested.as_nested_tensor(
        corrupted_rows,
        layout=torch.jagged,
    )
    materialized = materialize(
        pack_jagged_fields(
            wire,
            lengths=torch.ones(len(sample_ids), dtype=torch.long),
        ),
        layout="padded",
    )
    with pytest.raises(ValueError, match="payload digest"):
        decode_packed_tensor_wire(
            materialized,
            schema=schema,
            sample_ids=sample_ids,
        )


def test_packed_tensor_wire_rejects_partial_selection() -> None:
    batch, sample_ids = _media_batch()
    wire, schema = encode_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None
    first_entry = schema["entries"][0]
    partial = BatchedDataDict(
        {first_entry["present_field"]: wire[first_entry["present_field"]]}
    )
    with pytest.raises(ValueError, match="partially selected"):
        decode_packed_tensor_wire(
            partial,
            schema=schema,
            sample_ids=sample_ids,
        )


def test_media_wire_fields_are_not_padded_to_token_sequence_length() -> None:
    batch, sample_ids = _media_batch()
    wire, schema = encode_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None
    result = materialize(
        pack_jagged_fields(
            wire,
            lengths=torch.ones(len(sample_ids), dtype=torch.long),
        ),
        layout="padded",
        pad_to_seqlen=128,
    )
    for entry in schema["entries"]:
        assert result[entry["shapes_field"]].shape[1] == entry["rank"]
        assert result[entry["values_field"]].shape[1] < 128


def test_packed_tensor_schema_extends_lp_and_train_field_sets() -> None:
    batch, sample_ids = _media_batch()
    schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None
    fields = extend_fields_with_packed_tensor_wire(
        ["input_ids", "input_lengths", "token_mask", "sample_mask"],
        schema,
    )
    for field in packed_tensor_wire_field_names(schema):
        assert field in fields
    assert len(fields) == len(set(fields))


def test_tensorized_replica_components_round_trip_ragged_media() -> None:
    batch, _ = _media_batch()
    for logical_key in ("pixel_values", "imgs_sizes"):
        schema, components = packed_tensor_broadcast_components(
            logical_key,
            batch[logical_key],
        )
        decoded_key, decoded = packed_tensor_from_broadcast_components(
            schema,
            components,
        )
        assert decoded_key == logical_key
        _assert_same_packed(decoded, batch[logical_key])


def test_tensorized_replica_components_support_all_absent_dp_shard() -> None:
    batch, sample_ids = _media_batch()
    full_schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert full_schema is not None
    absent_schema = subset_packed_tensor_wire_schema(
        full_schema,
        sample_ids=[sample_ids[1]],
    )
    absent = batch["pixel_values"].slice([1])

    schema, components = packed_tensor_broadcast_components(
        "pixel_values",
        absent,
        expected_schema=absent_schema,
    )
    decoded_key, decoded = packed_tensor_from_broadcast_components(
        schema,
        components,
    )
    assert decoded_key == "pixel_values"
    assert decoded.tensors == [None]


def test_expected_schema_binds_exact_payload_before_first_put() -> None:
    batch, sample_ids = _media_batch()
    expected = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert expected is not None
    encode_packed_tensor_wire(
        batch,
        sample_ids=sample_ids,
        expected_schema=expected,
    )
    changed = deepcopy(batch)
    changed["pixel_values"].tensors[0] = changed["pixel_values"].tensors[0].clone()
    changed["pixel_values"].tensors[0][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="controller admission"):
        encode_packed_tensor_wire(
            changed,
            sample_ids=sample_ids,
            expected_schema=expected,
        )


def test_wire_schema_lives_only_in_meta_extra_info() -> None:
    batch, sample_ids = _media_batch()
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=["input_ids", "input_lengths", "sample_mask"],
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )
    meta = kv_first_write(
        batch,
        sample_ids=sample_ids,
        dp_client=client,
        partition_id="train",
    )
    assert PACKED_TENSOR_WIRE_SCHEMA_KEY in meta.extra_info
    assert PACKED_TENSOR_WIRE_SCHEMA_KEY not in (meta.fields or [])


def test_r3_trace_records_bfloat16_media_payload_and_schema(
    tmp_path,
    monkeypatch,
) -> None:
    batch, sample_ids = _media_batch()
    batch["pixel_values"] = PackedTensor(
        [
            row.to(torch.bfloat16) if row is not None else None
            for row in batch["pixel_values"].tensors
        ],
        dim_to_pack=batch["pixel_values"].dim_to_pack,
        pad_to_max_shape=batch["pixel_values"].pad_to_max_shape,
    )
    schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None

    from nemo_rl.utils import r3_trace

    monkeypatch.setenv("NRL_R3_TRACE", "1")
    monkeypatch.setenv("NRL_R3_TRACE_STEPS", "1")
    monkeypatch.setenv("NRL_R3_TRACE_SAMPLES", str(len(sample_ids)))
    monkeypatch.setenv("NRL_R3_TRACE_DIR", str(tmp_path))
    r3_trace._event_counts.clear()
    r3_trace.trace_tq_fetch_payload(
        stage="prev_lp",
        keys=sample_ids,
        data=batch,
        media_wire_schema_id=schema["wire_schema_id"],
    )

    records = [
        json.loads(line)
        for path in tmp_path.glob("r3_trace_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [record["key"] for record in records] == sample_ids
    assert all(
        record["media_wire_schema_id"] == schema["wire_schema_id"]
        for record in records
    )
    assert records[0]["packed_tensor_media"]["pixel_values"]["dtype"] == (
        "torch.bfloat16"
    )
    assert records[0]["packed_tensor_media"]["pixel_values"]["sha256"]
    r3_trace._event_counts.clear()


def test_dynamic_sampling_meta_concat_merges_media_row_authority() -> None:
    batch, sample_ids = _media_batch()
    first_ids = sample_ids[:2]
    second_ids = sample_ids[2:]
    first_batch = batch.slice(0, 2)
    second_batch = batch.slice(2, 3)
    first_schema = describe_packed_tensor_wire(first_batch, sample_ids=first_ids)
    second_schema = describe_packed_tensor_wire(second_batch, sample_ids=second_ids)
    assert first_schema is not None and second_schema is not None

    merged_schema = concat_packed_tensor_wire_schemas(
        [first_schema, second_schema],
        sample_id_groups=[first_ids, second_ids],
    )
    expected_schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert merged_schema == expected_schema

    from nemo_rl.data_plane.interfaces import KVBatchMeta

    first_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=first_ids,
        fields=packed_tensor_wire_field_names(first_schema),
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: first_schema},
    )
    second_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=second_ids,
        fields=packed_tensor_wire_field_names(second_schema),
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: second_schema},
    )
    merged_meta = first_meta.concat(second_meta)
    assert merged_meta.sample_ids == sample_ids
    assert merged_meta.extra_info[PACKED_TENSOR_WIRE_SCHEMA_KEY] == expected_schema


def test_dynamic_sampling_filter_then_concat_projects_media_authority() -> None:
    batch, sample_ids = _media_batch()
    schema = describe_packed_tensor_wire(batch, sample_ids=sample_ids)
    assert schema is not None

    from nemo_rl.data_plane.interfaces import KVBatchMeta

    full_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids,
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: schema},
    )
    survivors = full_meta.subset([0, 2])
    survivor_schema = survivors.extra_info[PACKED_TENSOR_WIRE_SCHEMA_KEY]
    assert survivor_schema == subset_packed_tensor_wire_schema(
        schema,
        sample_ids=[sample_ids[0], sample_ids[2]],
    )

    next_ids = ["sample-d"]
    next_batch = batch.slice(0, 1)
    next_schema = describe_packed_tensor_wire(next_batch, sample_ids=next_ids)
    assert next_schema is not None
    next_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=next_ids,
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: next_schema},
    )
    accumulated = survivors.concat(next_meta)
    assert accumulated.sample_ids == [sample_ids[0], sample_ids[2], "sample-d"]
    accumulated_schema = accumulated.extra_info[PACKED_TENSOR_WIRE_SCHEMA_KEY]
    assert accumulated_schema["sample_ids"] == accumulated.sample_ids
    validate_packed_tensor_wire_schema(
        accumulated_schema,
        expected_sample_ids=accumulated.sample_ids,
    )


def test_meta_concat_rejects_media_text_mix_and_structure_transition() -> None:
    batch, sample_ids = _media_batch()
    first_schema = describe_packed_tensor_wire(
        batch.slice(0, 2),
        sample_ids=sample_ids[:2],
    )
    assert first_schema is not None

    from nemo_rl.data_plane.interfaces import KVBatchMeta

    media_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids[:2],
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: first_schema},
    )
    text_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids[2:],
    )
    with pytest.raises(ValueError, match="mix media and text-only"):
        media_meta.concat(text_meta)

    changed_batch = batch.slice(2, 3)
    changed_batch["pixel_values"].dim_to_pack = 1
    changed_schema = describe_packed_tensor_wire(
        changed_batch,
        sample_ids=sample_ids[2:],
    )
    assert changed_schema is not None
    changed_meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids[2:],
        extra_info={PACKED_TENSOR_WIRE_SCHEMA_KEY: changed_schema},
    )
    with pytest.raises(ValueError, match="structure changed"):
        media_meta.concat(changed_meta)
