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

"""Dependency-light source invariants for the TQ exact-trace bridge."""

import ast
from pathlib import Path


_ROOT = Path(__file__).parents[3]


def _source(relative: str) -> str:
    return (_ROOT / relative).read_text()


def _function(relative: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    module = ast.parse(_source(relative), filename=relative)
    matches = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    assert len(matches) == 1, (relative, name, len(matches))
    return matches[0]


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [
        arg.arg
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
    ]


def test_scheduler_increment_crosses_tq_policy_and_worker_boundary():
    tq_train = _function("nemo_rl/models/policy/tq_policy.py", "train_from_meta")
    worker_train = _function(
        "nemo_rl/data_plane/worker_mixin.py",
        "train_presharded",
    )

    assert "scheduler_step_increment" in _arg_names(tq_train)
    assert "scheduler_step_increment" in _arg_names(worker_train)
    tq_text = ast.unparse(tq_train)
    worker_text = ast.unparse(worker_train)
    assert "scheduler_step_increment" in tq_text
    assert "train_kwargs['scheduler_step_increment']" in worker_text


def test_sync_exact_declares_physical_rows_without_grpo_grouping():
    helper = _function(
        "nemo_rl/algorithms/grpo_sync.py",
        "_execute_pending_exact_trace_step",
    )
    text = ast.unparse(helper)

    assert "num_samples=int(plan['total_row_count'])" in text
    assert "group_size=None" in text
    assert "gbs=int(plan['total_row_count'])" in text
    assert "scheduler_step_increment=int(plan['logical_rollout_count'])" in text
    assert "select_fields=['input_ids', 'content']" in text
    authority = _function(
        "nemo_rl/algorithms/grpo_sync.py",
        "_build_exact_trace_controller_authority",
    )
    authority_text = ast.unparse(authority)
    assert "pending_identity['optimizer_step_id'] != plan['optimizer_step_id']" in (
        authority_text
    )
    assert "pending_identity['logical_rollout_count']" in authority_text
    assert "pending_identity['group_size']" in authority_text


def test_exact_control_summary_does_not_carry_decoded_content():
    actor = _function(
        "nemo_rl/experience/sync_rollout_actor.py",
        "prepare_pending_exact_trace",
    )
    text = ast.unparse(actor)

    assert "train_data['content'] = np.asarray(content, dtype=object)" in text
    summary = next(
        node
        for node in ast.walk(actor)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "controller_summary"
    )
    assert isinstance(summary.value, ast.Dict)
    literal_keys = {
        key.value
        for key in summary.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "content" not in literal_keys


def test_sync_exact_has_non_idempotent_prepare_commit_abort_boundary():
    actor_source = _source("nemo_rl/experience/sync_rollout_actor.py")

    assert "def prepare_pending_exact_trace(" in actor_source
    assert "def commit_pending_exact_trace(" in actor_source
    assert "def abort_pending_exact_trace(" in actor_source
    assert "def finalize_pending_exact_trace(" in actor_source
    assert "refusing an implicit replay" in actor_source
    dispatch_at = actor_source.index(
        "self._exact_trace_failed_identity = pending_identity"
    )
    run_at = actor_source.index("result = run_nemo_gym_rollout_sync(", dispatch_at)
    publish_at = actor_source.index("self._exact_trace_pending = {", run_at)
    clear_at = actor_source.index(
        "self._exact_trace_failed_identity = None", publish_at
    )
    assert dispatch_at < run_at < publish_at < clear_at
    assert 'pending["commit_state"] = "committing"' in actor_source
    assert 'pending["intended_sample_ids"] = list(sample_ids)' in actor_source
    assert "Exact-trace TQ commit outcome is ambiguous" in actor_source
    assert "sample_ids=list(intended_sample_ids)" in actor_source
    assert 'sample_ids.append(f"{plan_id}:{row_index}")' in _source(
        "nemo_rl/experience/sync_exact_trace.py"
    )


def test_optimizer_boundary_is_structured_and_prompt_consumption_is_unchanged():
    sync_source = _source("nemo_rl/algorithms/grpo_sync.py")
    helper = _function(
        "nemo_rl/algorithms/grpo_sync.py",
        "_execute_pending_exact_trace_step",
    )
    helper_text = ast.unparse(helper)

    dispatched_at = helper_text.index("optimizer_update_dispatched = True")
    train_at = helper_text.index("train_results = policy.train_from_meta")
    applied_at = helper_text.index("optimizer_update_applied = True")
    finish_at = helper_text.index("policy.finish_step(meta)")
    assert dispatched_at < train_at < applied_at < finish_at
    assert "NRL_EXACT_TQ_STEP_PREPARED" in helper_text
    assert "NRL_EXACT_TQ_OPTIMIZER_DISPATCHED" in helper_text
    assert "NRL_EXACT_TQ_OPTIMIZER_APPLIED" in helper_text
    assert "NRL_EXACT_TQ_STEP_COMMITTED" in helper_text
    assert "NRL_EXACT_TQ_CHECKPOINT_SUBMITTED" in sync_source
    assert '"checkpoint_status": "finalization_submitted"' in sync_source
    assert "do not replay this optimizer_step_id" in helper_text
    assert "outcome is ambiguous" in helper_text
    assert "consumed_samples += master_config.grpo.num_prompts_per_step" in sync_source


def test_runtime_transfer_queue_pip_patch_is_absent_and_pin_is_gated():
    adapter = _source("nemo_rl/data_plane/adapters/transfer_queue.py")

    assert "_patch_tq_actor_runtime_env" not in adapter
    assert 'runtime_env = {"pip"' not in adapter
    assert (
        '_EXPECTED_TRANSFER_QUEUE_COMMIT = "b266d39a15aae114730de36cf8317b6285436f7f"'
        in adapter
    )
    assert "validate_baked_transfer_queue()" in adapter


def test_multimodal_exact_uses_first_class_wire_and_cp_replica_transport():
    actor = _source("nemo_rl/experience/sync_rollout_actor.py")
    worker = _source("nemo_rl/data_plane/worker_mixin.py")
    policy = _source("nemo_rl/models/policy/tq_policy.py")
    codec = _source("nemo_rl/data_plane/packed_tensor_wire.py")
    megatron_data = _source("nemo_rl/models/megatron/data.py")

    assert "TQ exact-trace PackedTensor transport is not implemented" not in actor
    assert "describe_packed_tensor_wire" in actor
    assert "packed_tensor_wire_schema" in actor
    assert "encode_packed_tensor_wire" in _source(
        "nemo_rl/data_plane/column_io.py"
    )
    assert "decode_packed_tensor_wire" in worker
    assert "packed_tensor_broadcast_components" in worker
    assert "extend_fields_with_packed_tensor_wire" in policy
    assert "TransferQueue/SingleController does not yet support models" not in worker
    assert "model_slices_context_parallel_inputs" in megatron_data
    assert "input_ids_cp_sharded = input_ids" in megatron_data
    assert "row_sha256_by_sample_id" in codec
    assert "broadcast_object_list" in worker
    assert "kind == \"packed_tensor\"" in worker
    sync_trainer = _source("nemo_rl/algorithms/grpo_sync.py")
    assert "extend_fields_with_packed_tensor_wire" in sync_trainer
    assert "packed_tensor_schema_from_extra_info(meta.extra_info)" in sync_trainer
    r3_trace = _source("nemo_rl/utils/r3_trace.py")
    assert 'record["media_wire_schema_id"]' in r3_trace
    assert 'record["packed_tensor_media"]' in r3_trace


def test_exact_logprob_writeback_is_canonical_and_not_upserted_twice():
    actor = _source("nemo_rl/experience/sync_rollout_actor.py")
    worker = _function(
        "nemo_rl/data_plane/worker_mixin.py",
        "_write_back_result_field",
    )
    worker_text = ast.unparse(worker)

    assert "effective_token_mask" in _arg_names(worker)
    assert "torch.isfinite" in worker_text
    assert "torch.where" in worker_text
    assert "skipped_fields" in actor
    assert "was not canonical" in actor
    assert "on masked token positions" in actor
