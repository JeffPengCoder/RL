# Recommended OSWorld training candidate for Omni v3

This candidate uses NeMo-RL with NeMo Gym and a caller-selected standard
Nemotron-3-Nano-Omni checkpoint. It does not use MOLT and does not require the
RFC0037 Step400 checkpoint. A concrete experiment should record its exact
checkpoint, image, dependency resolution, and dataset digest separately.

## Configuration

Use
[`grpo_nemotron_omni_30ba3b_osworld_recommended.yaml`](grpo_nemotron_omni_30ba3b_osworld_recommended.yaml)
for the four-node candidate. It adopts these recommendations:

| Setting | Candidate value |
|---|---:|
| Learning rate | `5e-6` |
| Router replay (R3) | enabled |
| Async GRPO | enabled, maximum trajectory age 1 |
| Token-level importance-sampling correction | enabled |
| GRPO leave-one-out baseline | enabled |
| Context length | 49,152 |
| Maximum OSWorld turns | 200 |
| Snapshot window | accumulate to 10, compact to 3 |
| Training topology | 2 nodes, TP2 / CP8 / EP8 |
| Rollout topology | 2 nodes, one TP8 replica per node |
| Prompt groups per step | 128 |
| Generations per prompt | 16 |

The exact sampled vLLM logprobs are retained by the trajectory contract and
the trainer reports `gen_kl_error` (also called vLLM-KL). No separate vLLM-KL
switch is required.

The suggested sequence-level `0.99-1.01` mask is intentionally not enabled in
this candidate. One logical rollout may materialize as several compacted
physical traces. The mask must first be implemented and tested at logical
`rollout_id` scope, then broadcast to every physical trace; applying the
existing per-sequence switch directly would change the optimization semantics.

## Data

Prefer an available, versioned Scale CUA manifest when its source and license
are accessible. Otherwise prepare all 361 standard OSWorld tasks as the
training manifest:

```bash
python examples/nemo_gym/prepare_osworld_exact_trace_data.py \
  --input <osworld-361.jsonl> \
  --expected-input-count 361 \
  --validation-count 0 \
  --train-output <osworld-361-exact-trace-train.jsonl>
```

The recommended training recipe disables in-run validation for this fallback,
so all 361 tasks remain training data. Evaluate independently with a genuinely
held-out manifest; do not silently reuse the training rows as validation.

## Launch boundary

Set at least:

```bash
export NANO_OMNI_MODEL_NAME=<standard-omni-v3-checkpoint>
export OSWORLD_TRAIN_DATA=<osworld-or-scalecua-exact-trace.jsonl>
export OSWORLD_CHECKPOINT_DIR=<writable-checkpoint-directory>
export OSWORLD_LOG_DIR=<writable-log-directory>
```

First rerun the short exact-trace recipe as a deployment canary. Then launch
the full candidate with four 8-GPU nodes. Exact model, image, Gym source,
dependency, and NCCL versions belong in that experiment's replay record; they
are not general support pins in this recipe.
