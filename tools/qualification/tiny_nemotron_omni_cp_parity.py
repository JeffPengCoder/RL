# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run a tiny Nemotron-Omni CP1/CP2 numerical parity qualification."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import socket
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NemotronOmniModelProvider,
)
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.packed_tensor_wire import describe_packed_tensor_wire
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.models.megatron.data import process_microbatch


IMAGE_TOKEN_ID = 18
RESULT_FORMAT = "b06-tiny-nemotron-omni-cp-parity-v1"


@dataclass
class TinyOmniProvider(NemotronOmniModelProvider):
    """Small real RADIO/Nemotron-H model used only for qualification."""

    has_sound: bool = False
    language_model_type: str = "nemotron6-moe"
    hidden_size: int = 128
    ffn_hidden_size: int = 256
    num_attention_heads: int = 4
    num_query_groups: int = 2
    kv_channels: int = 32
    mamba_num_heads: int = 4
    mamba_head_dim: int = 32
    mamba_num_groups: int = 2
    mamba_state_dim: int = 16
    hybrid_layer_pattern: str = "M"
    vocab_size: int = 128
    seq_length: int = 32
    image_token_index: int = IMAGE_TOKEN_ID
    img_start_token_id: int = 21
    img_end_token_id: int = 22
    tokenizer_type: str = "nemotron6-moe"
    dynamic_resolution: bool = True
    use_vision_backbone_fp8_arch: bool = False
    vision_proj_ffn_hidden_size: int = 256
    pipeline_model_parallel_size: int = 1
    use_cpu_initialization: bool = True
    gradient_accumulation_fusion: bool = False
    nemotron_omni_contract: str = NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT

    def _build_vision_config(self, language_cfg):
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        vision_cfg.mtp_num_layers = None
        vision_cfg.num_layers = 1
        vision_cfg.pipeline_model_parallel_size = 1
        vision_cfg.num_attention_heads = 4
        vision_cfg.add_bias_linear = True
        vision_cfg.add_qkv_bias = True
        vision_cfg.hidden_size = 128
        vision_cfg.ffn_hidden_size = 256
        vision_cfg.gated_linear_unit = False
        vision_cfg.kv_channels = 32
        vision_cfg.num_query_groups = 4
        vision_cfg.normalization = "LayerNorm"
        vision_cfg.qk_layernorm = False
        vision_cfg.layernorm_epsilon = 1e-6
        vision_cfg.class_token_len = 10
        return vision_cfg


def _module_path(module: Any) -> str:
    value = getattr(module, "__file__", None)
    if not value:
        raise RuntimeError(f"module {module.__name__} has no file identity")
    return str(Path(value).resolve(strict=True))


def _validate_provenance() -> dict[str, Any]:
    import megatron.bridge
    import megatron.core
    import nemo_rl
    import ray

    source_root = Path(os.environ["B06_SOURCE_ROOT"]).resolve(strict=True)
    actor_venv = Path(os.environ["B06_EXPECTED_ACTOR_VENV"]).resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != actor_venv:
        raise RuntimeError(
            f"wrong actor venv: prefix={sys.prefix}, expected={actor_venv}"
        )
    source_modules = (nemo_rl, megatron.bridge, megatron.core)
    dependency_modules = (torch, ray)
    source_paths = {module.__name__: _module_path(module) for module in source_modules}
    dependency_paths = {
        module.__name__: _module_path(module) for module in dependency_modules
    }
    for name, value in source_paths.items():
        if not Path(value).is_relative_to(source_root):
            raise RuntimeError(f"source module escaped mount: {name}={value}")
    for name, value in dependency_paths.items():
        if not Path(value).is_relative_to(actor_venv):
            raise RuntimeError(f"dependency escaped actor venv: {name}={value}")
    return {
        "actor_venv": str(actor_venv),
        "dependency_modules": dependency_paths,
        "executable": str(Path(sys.executable).resolve(strict=True)),
        "source_modules": source_paths,
        "source_root": str(source_root),
        "torch": torch.__version__,
    }


def _build_model(*, context_parallel_size: int):
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
    )
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    provider = TinyOmniProvider(
        freeze_language_model=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
        hybrid_layer_pattern="M",
    )
    provider.finalize()
    models = provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            check_for_nan_in_grad=True,
        ),
        wrap_with_ddp=True,
        mixed_precision_wrapper=None,
    )
    if len(models) != 1:
        raise RuntimeError(f"expected one model chunk, got {len(models)}")
    return models[0]


def _fixture(device: torch.device):
    input_ids = torch.tensor(
        [
            [7, 21, 18, 18, 22, 9, 10, 0],
            [11, 21, 18, 22, 12, 0, 0, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor([7, 5], dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    images = torch.randn(2, 3, 32, 64, generator=generator, device=device)
    images[1, :, :, 32:] = 0
    image_sizes = torch.tensor([[32, 64], [32, 32]], dtype=torch.int32, device=device)
    return input_ids, lengths, images, image_sizes


def _forward(model):
    device = torch.device("cuda", torch.cuda.current_device())
    input_ids, lengths, images, image_sizes = _fixture(device)
    processed = process_microbatch(
        {"input_ids": input_ids, "input_lengths": lengths},
        seq_length_key="input_lengths",
        pad_individual_seqs_to_multiple_of=4,
        pack_sequences=True,
        model_slices_context_parallel_inputs=True,
    )
    output = model(
        input_ids=processed.input_ids_cp_sharded,
        attention_mask=processed.attention_mask,
        packed_seq_params=processed.packed_seq_params,
        pixel_values=images,
        imgs_sizes=image_sizes,
    )
    logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        target=processed.input_ids,
        cu_seqlens_padded=processed.cu_seqlens_padded,
        unpacked_seqlen=input_ids.shape[1],
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        group=parallel_state.get_tensor_model_parallel_group(),
        inference_only=False,
        cp_group=parallel_state.get_context_parallel_group(),
    )
    prediction_mask = torch.arange(input_ids.shape[1] - 1, device=device).unsqueeze(
        0
    ) < (lengths - 1).unsqueeze(1)
    loss = -(logprobs * prediction_mask).sum() / prediction_mask.sum()
    media = {
        "pixel_values": PackedTensor([images[0:1], images[1:2]], dim_to_pack=0),
        "imgs_sizes": PackedTensor([image_sizes[0:1], image_sizes[1:2]], dim_to_pack=0),
    }
    media_schema = describe_packed_tensor_wire(
        media, sample_ids=["sample-0", "sample-1"]
    )
    if media_schema is None:
        raise RuntimeError("PackedTensor media schema was not generated")
    return loss, logprobs, media_schema


def _zero_stats() -> dict[str, float | int]:
    return {
        "changed_tensors": 0,
        "grad_abs_sum": 0.0,
        "grad_l2_sq": 0.0,
        "grad_max_abs": 0.0,
        "numel": 0,
        "update_abs_sum": 0.0,
        "update_l2_sq": 0.0,
        "update_max_abs": 0.0,
    }


def _parameter_group(name: str) -> str:
    if name.startswith("vision_model."):
        return "vision_model"
    if name.startswith("vision_projection."):
        return "vision_projection"
    raise RuntimeError(f"unexpected trainable parameter {name}")


def _train_and_summarize(model) -> tuple[dict[str, Any], torch.Tensor]:
    model.train()
    model.zero_grad_buffer()
    loss, logprobs, media_schema = _forward(model)
    loss.backward()
    model.finish_grad_sync()

    core_model = model.module
    groups = {
        "vision_model": _zero_stats(),
        "vision_projection": _zero_stats(),
    }
    before_update: dict[str, torch.Tensor] = {}
    optimizer_parameters = []
    for name, parameter in core_model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups[_parameter_group(name)]
        if not hasattr(parameter, "main_grad"):
            raise RuntimeError(f"missing main_grad for {name}")
        gradient = parameter.main_grad.detach()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite gradient for {name}")
        reference = gradient.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        group["grad_abs_sum"] += gradient.float().abs().sum().item()
        group["grad_l2_sq"] += gradient.float().square().sum().item()
        group["grad_max_abs"] = max(
            float(group["grad_max_abs"]), gradient.float().abs().max().item()
        )
        group["numel"] += gradient.numel()
        before_update[name] = parameter.detach().clone()
        parameter.grad = gradient.to(parameter.dtype).clone()
        optimizer_parameters.append(parameter)
    if not optimizer_parameters:
        raise RuntimeError("no trainable vision parameters were found")

    optimizer = torch.optim.SGD(optimizer_parameters, lr=0.01)
    optimizer.step()
    for name, parameter in core_model.named_parameters():
        if name not in before_update:
            continue
        update = before_update[name].float() - parameter.detach().float()
        group = groups[_parameter_group(name)]
        if torch.count_nonzero(update).item():
            group["changed_tensors"] += 1
        group["update_abs_sum"] += update.abs().sum().item()
        group["update_l2_sq"] += update.square().sum().item()
        group["update_max_abs"] = max(
            float(group["update_max_abs"]), update.abs().max().item()
        )
    for name, group in groups.items():
        if not group["changed_tensors"]:
            raise RuntimeError(f"optimizer did not update {name}")
        group["grad_l2"] = math.sqrt(float(group.pop("grad_l2_sq")))
        group["update_l2"] = math.sqrt(float(group.pop("update_l2_sq")))

    result = {
        "loss": loss.detach().float().item(),
        "logprobs": logprobs.detach().float().cpu().tolist(),
        "media_wire_schema_id": media_schema["wire_schema_id"],
        "media_wire_schema": media_schema,
        "parameter_groups": groups,
    }
    return result, logprobs


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _worker(
    rank: int,
    world_size: int,
    context_parallel_size: int,
    port: int,
    output_path: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=8),
    )
    model = None
    try:
        provenance = _validate_provenance()
        model = _build_model(context_parallel_size=context_parallel_size)
        result, logprobs = _train_and_summarize(model)
        reference = logprobs.detach().clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(logprobs, reference, rtol=0, atol=0)
        if rank == 0:
            _write_json_atomic(
                Path(output_path),
                {
                    "context_parallel_size": context_parallel_size,
                    "format": RESULT_FORMAT,
                    "provenance": provenance,
                    "result": result,
                    "world_size": world_size,
                },
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _run_mode(*, context_parallel_size: int, output: Path) -> None:
    if context_parallel_size not in (1, 2):
        raise ValueError("context parallel size must be 1 or 2")
    if torch.cuda.device_count() < context_parallel_size:
        raise RuntimeError(
            f"need {context_parallel_size} GPUs, got {torch.cuda.device_count()}"
        )
    if output.exists():
        raise FileExistsError(output)
    mp.spawn(
        _worker,
        args=(
            context_parallel_size,
            context_parallel_size,
            _free_port(),
            str(output),
        ),
        nprocs=context_parallel_size,
        join=True,
    )


def _assert_close(name: str, first: float, second: float) -> None:
    if not math.isclose(first, second, rel_tol=5e-3, abs_tol=5e-5):
        raise RuntimeError(f"{name} differs: cp1={first}, cp2={second}")


def _compare(cp1_path: Path, cp2_path: Path, output: Path) -> None:
    cp1 = json.loads(cp1_path.read_text(encoding="utf-8"))
    cp2 = json.loads(cp2_path.read_text(encoding="utf-8"))
    if cp1["format"] != RESULT_FORMAT or cp2["format"] != RESULT_FORMAT:
        raise RuntimeError("input result format changed")
    if cp1["context_parallel_size"] != 1 or cp2["context_parallel_size"] != 2:
        raise RuntimeError("CP result identities changed")
    first = cp1["result"]
    second = cp2["result"]
    if first["media_wire_schema_id"] != second["media_wire_schema_id"]:
        raise RuntimeError("CP1/CP2 media wire identities differ")
    _assert_close("loss", first["loss"], second["loss"])
    first_logprobs = torch.tensor(first["logprobs"], dtype=torch.float64)
    second_logprobs = torch.tensor(second["logprobs"], dtype=torch.float64)
    torch.testing.assert_close(first_logprobs, second_logprobs, rtol=5e-3, atol=5e-5)
    for group_name in ("vision_model", "vision_projection"):
        first_group = first["parameter_groups"][group_name]
        second_group = second["parameter_groups"][group_name]
        if first_group["numel"] != second_group["numel"]:
            raise RuntimeError(f"{group_name} parameter count differs")
        for metric in (
            "grad_abs_sum",
            "grad_l2",
            "grad_max_abs",
            "update_abs_sum",
            "update_l2",
            "update_max_abs",
        ):
            _assert_close(
                f"{group_name}.{metric}",
                first_group[metric],
                second_group[metric],
            )
    payload = {
        "cp1": str(cp1_path.resolve(strict=True)),
        "cp2": str(cp2_path.resolve(strict=True)),
        "format": RESULT_FORMAT,
        "loss_abs_diff": abs(first["loss"] - second["loss"]),
        "media_wire_schema_id": first["media_wire_schema_id"],
        "status": "passed",
    }
    _write_json_atomic(output, payload)
    print(
        "B06_TINY_NEMOTRON_OMNI_CP_PARITY|"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _run_all(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    cp1_path = output_dir / "cp1.json"
    cp2_path = output_dir / "cp2.json"
    comparison_path = output_dir / "comparison.json"
    _run_mode(context_parallel_size=1, output=cp1_path)
    _run_mode(context_parallel_size=2, output=cp2_path)
    _compare(cp1_path, cp2_path, comparison_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--context-parallel-size", type=int, choices=(1, 2), required=True
    )
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--cp1", type=Path, required=True)
    compare_parser.add_argument("--cp2", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "all":
        _run_all(args.output_dir)
        return
    if args.command == "run":
        _run_mode(
            context_parallel_size=args.context_parallel_size,
            output=args.output,
        )
        return
    _compare(args.cp1, args.cp2, args.output)


if __name__ == "__main__":
    main()
