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

"""First-class :class:`PackedTensor` transport for the TQ data plane.

``PackedTensor`` is a row-aligned list of optional, variably shaped tensors.
TransferQueue only transports TensorDict leaves, so every logical media key is
expanded into three tensor-only columns:

* a jagged flattened value row;
* an ``[N, rank]`` shape row (``-1`` for absent media);
* an ``[N]`` presence bit.

The primitive schema in ``KVBatchMeta.extra_info`` binds the logical key,
packing semantics, dtype, sample IDs, and per-row payload digests to those
synthetic columns.  Consumers reconstruct and rehash the ``PackedTensor``
before a policy/reference/train forward.  The same schema is also used by the
replica-leader tensor broadcast; media bytes never ride
``broadcast_object_list``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.schema import (
    PACKED_TENSOR_WIRE_FIELD_PREFIX,
    PACKED_TENSOR_WIRE_SCHEMA_KEY,
)


PACKED_TENSOR_WIRE_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str) or not name.startswith("torch."):
        raise ValueError(f"Invalid packed-tensor dtype name: {name!r}")
    dtype = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown packed-tensor dtype: {name!r}")
    return dtype


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to("cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    payload = {
        "dtype": _dtype_name(value.dtype),
        "shape": list(value.shape),
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return _digest(payload)


def _wire_field_names(logical_key: str) -> dict[str, str]:
    key_digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()[:24]
    stem = f"{PACKED_TENSOR_WIRE_FIELD_PREFIX}{key_digest}"
    return {
        "values_field": f"{stem}_values",
        "shapes_field": f"{stem}_shapes",
        "present_field": f"{stem}_present",
    }


def _validate_sample_ids(sample_ids: Sequence[str]) -> list[str]:
    result = list(sample_ids)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("PackedTensor wire sample IDs must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("PackedTensor wire sample IDs must be unique")
    return result


def _describe_entry(
    logical_key: str,
    value: PackedTensor,
    *,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(logical_key, str) or not logical_key:
        raise ValueError("PackedTensor logical keys must be non-empty strings")
    ids = _validate_sample_ids(sample_ids)
    if len(value) != len(ids):
        raise ValueError(
            f"PackedTensor {logical_key!r} has {len(value)} rows, expected {len(ids)}"
        )
    present = [item for item in value.tensors if item is not None]
    if not present:
        raise ValueError(
            f"PackedTensor {logical_key!r} has no present row from which to bind "
            "dtype and rank"
        )
    if any(not isinstance(item, torch.Tensor) for item in present):
        raise TypeError(f"PackedTensor {logical_key!r} contains a non-tensor row")
    dtype = present[0].dtype
    rank = present[0].ndim
    if rank <= 0:
        raise ValueError(f"PackedTensor {logical_key!r} rows must have rank >= 1")
    if any(item.dtype != dtype or item.ndim != rank for item in present):
        raise ValueError(
            f"PackedTensor {logical_key!r} rows must share one dtype and rank"
        )
    pack_dim = value.dim_to_pack if value.dim_to_pack >= 0 else rank + value.dim_to_pack
    if not 0 <= pack_dim < rank:
        raise ValueError(
            f"PackedTensor {logical_key!r} dim_to_pack={value.dim_to_pack} is "
            f"invalid for rank {rank}"
        )

    row_sha256_by_sample_id: dict[str, str | None] = {}
    row_numel_by_sample_id: dict[str, int] = {}
    total_value_count = 0
    for sample_id, item in zip(ids, value.tensors, strict=True):
        if item is None:
            row_sha256_by_sample_id[sample_id] = None
            row_numel_by_sample_id[sample_id] = 0
            continue
        item = item.detach().to("cpu").contiguous()
        row_sha256_by_sample_id[sample_id] = _tensor_sha256(item)
        row_numel_by_sample_id[sample_id] = item.numel()
        total_value_count += item.numel()

    entry: dict[str, Any] = {
        "logical_key": logical_key,
        **_wire_field_names(logical_key),
        "dtype": _dtype_name(dtype),
        "rank": rank,
        "dim_to_pack": int(value.dim_to_pack),
        "pad_to_max_shape": bool(value.pad_to_max_shape),
        "row_sha256_by_sample_id": row_sha256_by_sample_id,
        "row_numel_by_sample_id": row_numel_by_sample_id,
        "total_value_count": total_value_count,
    }
    entry["payload_sha256"] = _digest(entry)
    return entry


def describe_packed_tensor_wire(
    data: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Return the primitive authority for every PackedTensor in ``data``."""
    ids = _validate_sample_ids(sample_ids)
    entries = [
        _describe_entry(key, value, sample_ids=ids)
        for key, value in sorted(data.items())
        if isinstance(value, PackedTensor)
    ]
    if not entries:
        return None
    schema: dict[str, Any] = {
        "schema_version": PACKED_TENSOR_WIRE_SCHEMA_VERSION,
        "sample_ids": ids,
        "entries": entries,
    }
    schema["wire_schema_id"] = _digest(schema)
    return validate_packed_tensor_wire_schema(schema, expected_sample_ids=ids)


def validate_packed_tensor_wire_schema(
    schema: Mapping[str, Any],
    *,
    expected_sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate and return a JSON-canonical copy of a media wire schema."""
    if not isinstance(schema, Mapping):
        raise TypeError("PackedTensor wire schema must be a mapping")
    normalized = json.loads(json.dumps(schema))
    if normalized.get("schema_version") != PACKED_TENSOR_WIRE_SCHEMA_VERSION:
        raise ValueError("Unsupported PackedTensor wire schema version")
    sample_ids = _validate_sample_ids(normalized.get("sample_ids", []))
    if expected_sample_ids is not None and sample_ids != list(expected_sample_ids):
        raise ValueError("PackedTensor wire schema sample IDs changed")
    entries = normalized.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("PackedTensor wire schema must contain entries")

    logical_keys: set[str] = set()
    wire_fields: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("PackedTensor wire entry must be a mapping")
        logical_key = entry.get("logical_key")
        if not isinstance(logical_key, str) or not logical_key:
            raise ValueError("PackedTensor wire logical_key is invalid")
        if logical_key in logical_keys:
            raise ValueError(f"Duplicate PackedTensor logical key {logical_key!r}")
        logical_keys.add(logical_key)
        expected_fields = _wire_field_names(logical_key)
        for field_key, expected in expected_fields.items():
            if entry.get(field_key) != expected:
                raise ValueError(
                    f"PackedTensor wire field {field_key!r} changed for {logical_key!r}"
                )
            if expected in wire_fields:
                raise ValueError(f"Duplicate PackedTensor wire field {expected!r}")
            wire_fields.add(expected)
        _dtype_from_name(entry.get("dtype"))
        rank = entry.get("rank")
        dim_to_pack = entry.get("dim_to_pack")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("PackedTensor wire rank must be positive")
        if isinstance(dim_to_pack, bool) or not isinstance(dim_to_pack, int):
            raise ValueError("PackedTensor wire dim_to_pack must be an integer")
        normalized_dim = dim_to_pack if dim_to_pack >= 0 else rank + dim_to_pack
        if not 0 <= normalized_dim < rank:
            raise ValueError("PackedTensor wire dim_to_pack is out of range")
        if not isinstance(entry.get("pad_to_max_shape"), bool):
            raise ValueError("PackedTensor wire pad_to_max_shape must be boolean")
        row_digests = entry.get("row_sha256_by_sample_id")
        row_numel = entry.get("row_numel_by_sample_id")
        if not isinstance(row_digests, dict) or set(row_digests) != set(sample_ids):
            raise ValueError("PackedTensor row digest authority is incomplete")
        if not isinstance(row_numel, dict) or set(row_numel) != set(sample_ids):
            raise ValueError("PackedTensor row size authority is incomplete")
        total = 0
        for sample_id in sample_ids:
            digest = row_digests[sample_id]
            count = row_numel[sample_id]
            if digest is not None and (
                not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None
            ):
                raise ValueError("PackedTensor row digest is invalid")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("PackedTensor row element count is invalid")
            if (digest is None) != (count == 0):
                # A present zero-sized tensor is deliberately rejected: it is
                # indistinguishable from absence on several TQ backends.
                raise ValueError("PackedTensor absence and row size disagree")
            total += count
        if entry.get("total_value_count") != total:
            raise ValueError("PackedTensor total element count is corrupted")
        payload_sha256 = entry.get("payload_sha256")
        entry_without_digest = dict(entry)
        entry_without_digest.pop("payload_sha256", None)
        if payload_sha256 != _digest(entry_without_digest):
            raise ValueError("PackedTensor payload authority digest is corrupted")

    wire_schema_id = normalized.get("wire_schema_id")
    schema_without_digest = dict(normalized)
    schema_without_digest.pop("wire_schema_id", None)
    if wire_schema_id != _digest(schema_without_digest):
        raise ValueError("PackedTensor wire schema digest is corrupted")
    return normalized


def packed_tensor_wire_field_names(schema: Mapping[str, Any] | None) -> list[str]:
    if schema is None:
        return []
    validated = validate_packed_tensor_wire_schema(schema)
    return [
        entry[field]
        for entry in validated["entries"]
        for field in ("values_field", "shapes_field", "present_field")
    ]


def extend_fields_with_packed_tensor_wire(
    fields: Sequence[str],
    schema: Mapping[str, Any] | None,
) -> list[str]:
    result = list(fields)
    for field in packed_tensor_wire_field_names(schema):
        if field not in result:
            result.append(field)
    return result


def packed_tensor_schema_from_extra_info(
    extra_info: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not extra_info or PACKED_TENSOR_WIRE_SCHEMA_KEY not in extra_info:
        return None
    return validate_packed_tensor_wire_schema(extra_info[PACKED_TENSOR_WIRE_SCHEMA_KEY])


def concat_packed_tensor_wire_schemas(
    schemas: Sequence[Mapping[str, Any]],
    *,
    sample_id_groups: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Merge row authorities when dynamic sampling concatenates TQ metas.

    Every rollout iteration owns a distinct set of sample IDs and therefore a
    distinct wire-schema digest, even when the processor emits the same media
    keys.  :meth:`KVBatchMeta.concat` must merge those row authorities rather
    than retaining the first iteration's schema.  Structural media properties
    (logical keys, dtype, rank, packing dimension and synthetic field names)
    must remain identical across iterations; a processor/schema transition in
    the middle of one optimizer step fails closed.
    """
    if not schemas or len(schemas) != len(sample_id_groups):
        raise ValueError(
            "PackedTensor schema concat requires one schema per sample-ID group"
        )

    validated = [
        validate_packed_tensor_wire_schema(
            schema,
            expected_sample_ids=sample_ids,
        )
        for schema, sample_ids in zip(schemas, sample_id_groups, strict=True)
    ]
    merged_sample_ids = [sample_id for group in sample_id_groups for sample_id in group]
    _validate_sample_ids(merged_sample_ids)

    first_entries = {entry["logical_key"]: entry for entry in validated[0]["entries"]}
    expected_keys = set(first_entries)
    structural_fields = (
        "logical_key",
        "values_field",
        "shapes_field",
        "present_field",
        "dtype",
        "rank",
        "dim_to_pack",
        "pad_to_max_shape",
    )
    for schema in validated[1:]:
        if {entry["logical_key"] for entry in schema["entries"]} != expected_keys:
            raise ValueError(
                "PackedTensor logical keys changed across dynamic-sampling batches"
            )
        current = {entry["logical_key"]: entry for entry in schema["entries"]}
        for logical_key in expected_keys:
            if any(
                current[logical_key][field] != first_entries[logical_key][field]
                for field in structural_fields
            ):
                raise ValueError(
                    "PackedTensor structure changed across dynamic-sampling "
                    f"batches for {logical_key!r}"
                )

    merged_entries: list[dict[str, Any]] = []
    for logical_key in sorted(expected_keys):
        merged = {
            field: first_entries[logical_key][field] for field in structural_fields
        }
        row_sha256_by_sample_id: dict[str, str | None] = {}
        row_numel_by_sample_id: dict[str, int] = {}
        for schema in validated:
            entry = next(
                item for item in schema["entries"] if item["logical_key"] == logical_key
            )
            row_sha256_by_sample_id.update(entry["row_sha256_by_sample_id"])
            row_numel_by_sample_id.update(entry["row_numel_by_sample_id"])
        merged["row_sha256_by_sample_id"] = row_sha256_by_sample_id
        merged["row_numel_by_sample_id"] = row_numel_by_sample_id
        merged["total_value_count"] = sum(row_numel_by_sample_id.values())
        merged["payload_sha256"] = _digest(merged)
        merged_entries.append(merged)

    result: dict[str, Any] = {
        "schema_version": PACKED_TENSOR_WIRE_SCHEMA_VERSION,
        "sample_ids": merged_sample_ids,
        "entries": merged_entries,
    }
    result["wire_schema_id"] = _digest(result)
    return validate_packed_tensor_wire_schema(
        result,
        expected_sample_ids=merged_sample_ids,
    )


def subset_packed_tensor_wire_schema(
    schema: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Project a full media authority onto an ordered metadata subset."""
    validated = validate_packed_tensor_wire_schema(schema)
    ids = _validate_sample_ids(sample_ids)
    authority_ids = set(validated["sample_ids"])
    if any(sample_id not in authority_ids for sample_id in ids):
        raise ValueError("PackedTensor subset requested an unknown sample ID")

    structural_fields = (
        "logical_key",
        "values_field",
        "shapes_field",
        "present_field",
        "dtype",
        "rank",
        "dim_to_pack",
        "pad_to_max_shape",
    )
    entries: list[dict[str, Any]] = []
    for entry in validated["entries"]:
        projected = {field: entry[field] for field in structural_fields}
        projected["row_sha256_by_sample_id"] = {
            sample_id: entry["row_sha256_by_sample_id"][sample_id] for sample_id in ids
        }
        projected["row_numel_by_sample_id"] = {
            sample_id: entry["row_numel_by_sample_id"][sample_id] for sample_id in ids
        }
        projected["total_value_count"] = sum(
            projected["row_numel_by_sample_id"].values()
        )
        projected["payload_sha256"] = _digest(projected)
        entries.append(projected)

    result: dict[str, Any] = {
        "schema_version": PACKED_TENSOR_WIRE_SCHEMA_VERSION,
        "sample_ids": ids,
        "entries": entries,
    }
    result["wire_schema_id"] = _digest(result)
    return validate_packed_tensor_wire_schema(result, expected_sample_ids=ids)


def encode_packed_tensor_wire(
    data: Mapping[str, Any],
    *,
    sample_ids: Sequence[str],
    expected_schema: Mapping[str, Any] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Encode all PackedTensor values into synthetic tensor-only columns."""
    schema = describe_packed_tensor_wire(data, sample_ids=sample_ids)
    if expected_schema is not None:
        expected = validate_packed_tensor_wire_schema(
            expected_schema,
            expected_sample_ids=sample_ids,
        )
        if schema != expected:
            raise ValueError("PackedTensor payload changed after controller admission")
    if schema is None:
        return {}, None

    wire: dict[str, torch.Tensor] = {}
    for entry in schema["entries"]:
        packed = data[entry["logical_key"]]
        assert isinstance(packed, PackedTensor)
        dtype = _dtype_from_name(entry["dtype"])
        rows: list[torch.Tensor] = []
        shapes = torch.full(
            (len(sample_ids), int(entry["rank"])),
            -1,
            dtype=torch.int64,
        )
        present = torch.zeros(len(sample_ids), dtype=torch.bool)
        for row_index, item in enumerate(packed.tensors):
            if item is None:
                rows.append(torch.empty(0, dtype=dtype))
                continue
            item = item.detach().to("cpu").contiguous()
            if item.numel() == 0:
                raise ValueError("Present PackedTensor rows must not be empty")
            rows.append(item.reshape(-1))
            shapes[row_index] = torch.tensor(item.shape, dtype=torch.int64)
            present[row_index] = True
        wire[entry["values_field"]] = torch.nested.as_nested_tensor(
            rows,
            layout=torch.jagged,
        )
        wire[entry["shapes_field"]] = shapes
        wire[entry["present_field"]] = present
    return wire, schema


def _row_from_wire_values(
    values: torch.Tensor, row_index: int, rows: int
) -> torch.Tensor:
    if values.is_nested:
        return values[row_index].reshape(-1)
    if values.ndim == 1:
        if rows == 1:
            return values.reshape(-1)
        if values.shape[0] != rows:
            raise ValueError("PackedTensor wire values lost their batch dimension")
        return values[row_index : row_index + 1]
    if values.shape[0] != rows:
        raise ValueError("PackedTensor wire values have the wrong batch size")
    return values[row_index].reshape(-1)


def decode_packed_tensor_wire(
    data: MutableMapping[str, Any],
    *,
    schema: Mapping[str, Any] | None,
    sample_ids: Sequence[str],
) -> MutableMapping[str, Any]:
    """Reconstruct and rehash PackedTensor values selected from TQ."""
    if schema is None:
        return data
    validated = validate_packed_tensor_wire_schema(schema)
    ids = _validate_sample_ids(sample_ids)
    authority_ids = validated["sample_ids"]
    if any(sample_id not in authority_ids for sample_id in ids):
        raise ValueError("PackedTensor consumer requested an unknown sample ID")

    for entry in validated["entries"]:
        fields = [
            entry["values_field"],
            entry["shapes_field"],
            entry["present_field"],
        ]
        selected = [field in data for field in fields]
        if not any(selected):
            continue
        if not all(selected):
            raise ValueError(
                f"PackedTensor logical key {entry['logical_key']!r} was only "
                "partially selected from TQ"
            )
        if entry["logical_key"] in data:
            raise ValueError("PackedTensor logical key collides with a wire column")

        values = data.pop(entry["values_field"])
        shapes = data.pop(entry["shapes_field"])
        present = data.pop(entry["present_field"])
        if not all(
            isinstance(item, torch.Tensor) for item in (values, shapes, present)
        ):
            raise TypeError("PackedTensor wire columns must all be tensors")
        rows = len(ids)
        rank = int(entry["rank"])
        if shapes.ndim == 1 and rank == 1 and shapes.shape[0] == rows:
            shapes = shapes.unsqueeze(-1)
        if (
            shapes.dtype != torch.int64
            or shapes.ndim != 2
            or shapes.shape[0] != rows
            or shapes.shape[1] < rank
        ):
            raise ValueError("PackedTensor shape rows are malformed")
        if shapes.shape[1] > rank and torch.count_nonzero(shapes[:, rank:]).item() != 0:
            raise ValueError("PackedTensor padded shape tail is nonzero")
        if present.ndim == 2 and present.shape == (rows, 1):
            present = present.squeeze(-1)
        if present.dtype != torch.bool or present.shape != (rows,):
            raise ValueError("PackedTensor presence rows are malformed")
        if values.dtype != _dtype_from_name(entry["dtype"]):
            raise ValueError("PackedTensor wire dtype changed")

        tensors: list[torch.Tensor | None] = []
        for row_index, sample_id in enumerate(ids):
            shape_values = [int(item) for item in shapes[row_index, :rank].tolist()]
            expected_digest = entry["row_sha256_by_sample_id"][sample_id]
            expected_numel = int(entry["row_numel_by_sample_id"][sample_id])
            is_present = bool(present[row_index].item())
            flat = _row_from_wire_values(values, row_index, rows)
            if not is_present:
                if expected_digest is not None or expected_numel != 0:
                    raise ValueError("PackedTensor presence bit changed")
                if any(size != -1 for size in shape_values):
                    raise ValueError("Absent PackedTensor row has a concrete shape")
                tensors.append(None)
                continue
            if expected_digest is None or expected_numel <= 0:
                raise ValueError("PackedTensor presence bit changed")
            if any(size < 0 for size in shape_values):
                raise ValueError("Present PackedTensor row has an invalid shape")
            numel = math.prod(shape_values)
            if numel != expected_numel or flat.numel() < numel:
                raise ValueError("PackedTensor row shape/size authority changed")
            if flat.numel() > numel and torch.count_nonzero(flat[numel:]).item() != 0:
                raise ValueError("PackedTensor padded wire tail is nonzero")
            tensor = flat[:numel].reshape(shape_values).contiguous()
            if _tensor_sha256(tensor) != expected_digest:
                raise ValueError("PackedTensor row payload digest changed")
            tensors.append(tensor)
        data[entry["logical_key"]] = PackedTensor(
            tensors,
            dim_to_pack=int(entry["dim_to_pack"]),
            pad_to_max_shape=bool(entry["pad_to_max_shape"]),
        )
    return data


def packed_tensor_broadcast_components(
    logical_key: str,
    value: PackedTensor,
    *,
    expected_schema: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Flatten one PackedTensor into tensors suitable for NCCL/Gloo broadcast."""
    if expected_schema is None:
        sample_ids = [f"replica-row-{index}" for index in range(len(value))]
        schema = describe_packed_tensor_wire(
            {logical_key: value},
            sample_ids=sample_ids,
        )
        assert schema is not None
    else:
        full_schema = validate_packed_tensor_wire_schema(expected_schema)
        sample_ids = full_schema["sample_ids"]
        if len(value) != len(sample_ids):
            raise ValueError("PackedTensor replica rows changed after TQ decode")
        entries = [
            item
            for item in full_schema["entries"]
            if item["logical_key"] == logical_key
        ]
        if len(entries) != 1:
            raise ValueError(
                f"PackedTensor replica schema does not bind {logical_key!r}"
            )
        entry = entries[0]
        if any(item is not None for item in value.tensors):
            if _describe_entry(logical_key, value, sample_ids=sample_ids) != entry:
                raise ValueError("PackedTensor replica payload changed after TQ decode")
        elif any(
            entry["row_sha256_by_sample_id"][sample_id] is not None
            or entry["row_numel_by_sample_id"][sample_id] != 0
            for sample_id in sample_ids
        ):
            raise ValueError("PackedTensor replica presence changed after TQ decode")
        schema = {
            "schema_version": PACKED_TENSOR_WIRE_SCHEMA_VERSION,
            "sample_ids": sample_ids,
            "entries": [entry],
        }
        schema["wire_schema_id"] = _digest(schema)
        schema = validate_packed_tensor_wire_schema(
            schema,
            expected_sample_ids=sample_ids,
        )
    entry = schema["entries"][0]
    dtype = _dtype_from_name(entry["dtype"])
    flat_rows: list[torch.Tensor] = []
    shapes = torch.full((len(value), entry["rank"]), -1, dtype=torch.int64)
    present = torch.zeros(len(value), dtype=torch.bool)
    offsets = [0]
    for row_index, item in enumerate(value.tensors):
        if item is None:
            offsets.append(offsets[-1])
            continue
        row = item.detach().to("cpu").contiguous()
        flat_rows.append(row.reshape(-1))
        shapes[row_index] = torch.tensor(row.shape, dtype=torch.int64)
        present[row_index] = True
        offsets.append(offsets[-1] + row.numel())
    values = torch.cat(flat_rows) if flat_rows else torch.empty(0, dtype=dtype)
    return schema, (
        values,
        torch.tensor(offsets, dtype=torch.int64),
        shapes,
        present,
    )


def packed_tensor_from_broadcast_components(
    schema: Mapping[str, Any],
    components: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[str, PackedTensor]:
    """Reconstruct one PackedTensor after tensorized replica broadcast."""
    validated = validate_packed_tensor_wire_schema(schema)
    if len(validated["entries"]) != 1:
        raise ValueError("Replica PackedTensor schema must contain one logical key")
    entry = validated["entries"][0]
    values, offsets, shapes, present = components
    sample_ids = validated["sample_ids"]
    rows = len(sample_ids)
    if offsets.shape != (rows + 1,) or offsets.dtype != torch.int64:
        raise ValueError("PackedTensor broadcast offsets are malformed")
    if values.dtype != _dtype_from_name(entry["dtype"]):
        raise ValueError("PackedTensor broadcast value dtype changed")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != values.numel():
        raise ValueError("PackedTensor broadcast offsets changed")
    if torch.any(offsets[1:] < offsets[:-1]):
        raise ValueError("PackedTensor broadcast offsets are not monotonic")
    rank = int(entry["rank"])
    if (
        shapes.shape != (rows, rank)
        or shapes.dtype != torch.int64
        or present.shape != (rows,)
        or present.dtype != torch.bool
    ):
        raise ValueError("PackedTensor broadcast metadata shape changed")

    tensors: list[torch.Tensor | None] = []
    for row_index, sample_id in enumerate(sample_ids):
        start = int(offsets[row_index].item())
        end = int(offsets[row_index + 1].item())
        expected_digest = entry["row_sha256_by_sample_id"][sample_id]
        if not bool(present[row_index].item()):
            if start != end or expected_digest is not None:
                raise ValueError("PackedTensor broadcast absence changed")
            tensors.append(None)
            continue
        shape = [int(item) for item in shapes[row_index].tolist()]
        if any(size < 0 for size in shape) or math.prod(shape) != end - start:
            raise ValueError("PackedTensor broadcast shape changed")
        tensor = values[start:end].reshape(shape).contiguous()
        if _tensor_sha256(tensor) != expected_digest:
            raise ValueError("PackedTensor broadcast payload digest changed")
        tensors.append(tensor)
    return entry["logical_key"], PackedTensor(
        tensors,
        dim_to_pack=int(entry["dim_to_pack"]),
        pad_to_max_shape=bool(entry["pad_to_max_shape"]),
    )
