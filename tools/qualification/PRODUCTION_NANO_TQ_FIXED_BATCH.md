# Production Nano-Omni fixed-batch TQ qualification

## Scope and runtime authority

This lane proves one real Nano-Omni Megatron update through the production
TransferQueue data path without involving rollout generation, Gym, OSWorld, or
vLLM:

```text
processor-created image/text batch
  -> TQ prepare_step
  -> TQ kv_first_write
  -> current-policy logprobs
  -> reference-policy logprobs
  -> optimizer update
  -> checkpoint + evidence join
```

The runtime baseline is the official amd64 NGC release
`nvcr.io/nvidia/nemo-rl:v0.7.0` at the scheduler-approved immutable digest.
The earlier private r1m assumptions (vLLM 0.25.1 and Ray 2.56.1) are
superseded. The image supplies software and native ABI dependencies only. One
complete manifest-qualified source bundle is mounted read-only at
`/workspace/source`; output, journals, R3 records, and checkpoints go to a
fresh Lustre attempt directory.

The abbreviated digest `sha256:9392cc2...e4f9` is documentation, not a valid
launch authority. A launcher must receive and rehash the full immutable image
digest before this lane runs.

The official release matrix is documented in the
[NeMo-RL v0.7.0 release](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.7.0),
and the image/source-mount model is documented in the
[NeMo-RL installation guide](https://docs.nvidia.com/nemo/rl/latest/about/installation.html).

## Compatibility audit against v0.7.0

| Boundary | Official v0.7.0 | Mounted successor source | Decision / required gate |
| --- | --- | --- | --- |
| Python | 3.13.13 | `.python-version=3.13.14`, `requires-python>=3.13.14` | **P0 mismatch.** Do not call the mounted stack release-qualified until a bounded source metadata compatibility delta is reviewed and the entire driver/actor suite passes under 3.13.13, or the runtime supplies 3.13.14. Runtime execution that happens to work is not enough. |
| PyTorch | 2.11.0 | exactly 2.11.0 | Compatible in version; still prove CUDA, TE, NCCL, and compiled-extension imports in every actor venv. |
| Ray | 2.55.1 | declares `ray[default]>=2.55.1`; the newer lock happened to select 2.56.1 | 2.55.1 satisfies the source contract. Require one version across driver, TQ controller/storage, and every Megatron actor, plus real placement-group and `runtime_env.py_executable` smoke. |
| TransferQueue | commit `b266d39` | same required commit | Good dependency match. Rehash `direct_url.json`, import path, controller actor, storage actor, simple backend, and Mooncake backend rather than assuming the package is present in every prefetched venv. |
| Mooncake | CUDA-13 wheel `0.3.11.post1` | same required wheel | Good declared match. Require `mooncake.store` import, `mooncake_master` executable, CUDA-13 shared-library closure, and real cross-process put/get. |
| TensorDict | installed by the data-plane base dependency | successor media codec depends on TensorDict leaf and slicing behavior | Record the exact installed version and run PackedTensor simple/Mooncake round trips. Do not infer compatibility only from import success. |
| vLLM | 0.20.0 | successor lock pins 0.25.1 | Not used by this fixed-batch lane. It remains a separate P0 qualification for the full OSWorld rollout/training chain; this harness must never be cited as vLLM compatibility evidence. |
| NeMo Gym | `0.4.0+d67ad66` | exact-trace source uses Gym `f799a54` | Not used by this lane. Full OSWorld training must mount and attest `f799a54` (or a reviewed successor) and qualify its server venvs. The stock release does not prove those exact-trace contracts. |
| Megatron-Bridge | `0.5.0+554c7b9` | source bundle pins `573e088c` | Pure Python must resolve from `/workspace/source`, while native dependencies remain image-owned. Run model import, HF-to-MCore conversion, prev/ref/train, optimizer, and checkpoint gates; the commit difference cannot be waived. |
| Megatron-Core | release submodule via Bridge | source bundle's recursive Bridge submodule | Same rule as Bridge: mounted source identity plus image-owned native ABI, each independently attested. |
| Transformer Engine | image-selected Torch 2.11 / CUDA 13 build (not versioned in the release component table) | mounted source and Bridge select their own Python integration | Do not infer an exact TE version from the release label. Record package/build identity and shared-library closure in the actor, then require a real Nano forward/backward on H100. |
| Prefetched worker venv | image-owned under `/opt/ray_venvs` and built from v0.7 source | actor imports successor source | Reuse only the dependency environment. Scheduler sets the exact source roots; actor provenance must show `nemo_rl`, `megatron.bridge`, and `megatron.core` at their canonical files below `/workspace/source`. Torch/Ray/TQ must equal the top-level package anchors selected by the exact actor interpreter's distribution metadata; this admits content-identical uv links into an image-owned cache without accepting unrelated files. |

The official v0.7 Dockerfile prefetches worker environments, and its root
dependency set already contains TransferQueue and Mooncake. That makes the
official image a plausible native/runtime substrate. It does **not** make its
editable `/opt/nemo-rl` source or its older Gym/Bridge commits authoritative
for this experiment.

The mounted successor also declares `nixl`, `awscrt`, `zstandard`, and
`fastokens-b10` beyond the fixed-batch core. They are not exercised by this
lane when checkpoint-engine/NIXL refit, sparse S3/ZMQ refit, and Fastokens are
all disabled: their imports are configuration-gated or confined to those
subsystems. Missing them therefore need not block this one fixed-batch gate,
but it **does** block any later claim covering the full rollout/refit path.
Conversely, `torch`, `ray`, `tensordict`, TransferQueue, Mooncake (for the
production backend), Pillow, Transformers/processor dependencies,
OmegaConf/Hydra, Megatron-Bridge/Core, and Transformer Engine are on the
fixed-batch execution path and must be proven in the exact driver/actor venvs.

Before allocating the 30B model, run an image-only compatibility probe in each
selected interpreter. It must record `sys.prefix`, `sys.executable`, Python,
Ray, Torch/CUDA, TensorDict, TransferQueue direct URL/commit, Mooncake wheel and
binary, Megatron-Bridge/Core module realpaths, Transformer Engine import and
shared-library closure, plus the mounted `nemo_rl` realpath. No package install
or runtime `pip` environment is permitted during this probe.

The scheduler must also run the separately owned full source-tree verifier and
HF snapshot verifier inside the container immediately before this entry point.
This harness rehashes and records their selected identity/model manifests and
then proves imported module realpaths; it does not replace their all-files,
no-extra, symlink, dynamic-module, or safetensors checks.

## Source and mount layout

```text
/workspace/source:ro
  nemo_rl/
  tools/qualification/
  3rdparty/Gym-workspace/Gym/
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/
  SOURCE_BUNDLE.json

/workspace/model:ro
  complete Nano-Omni snapshot + dynamic processor modules + MODEL_MANIFEST

/workspace/output:rw
  production-nano-tq/<fresh-run-id>/
    evidence/
    journal/
    r3/
    checkpoint/
```

The driver and actors use this exact source path set, in order:

```text
/workspace/source
/workspace/source/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src
/workspace/source/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
```

It is scheduler-owned and hash-bound to `SOURCE_BUNDLE.json`; a caller cannot
append arbitrary `PYTHONPATH` entries. No host venv, `.git`, source patch, or
individual Python-file overlay is allowed.

## CP1 production lane

Use one H100 node with eight GPUs and a resolved configuration equivalent to:

```text
world size = 8
TP = 8
PP = 1
CP = 1
DP = 1
EP = 8 (model recipe setting)
global fixed batch = 8
micro batch = 1
router replay = false
dynamic batching = false
sequence packing = false
Fastokens = false (and NRL_USE_FASTOKENS absent or 0)
generation refit transport = null
reference-policy KL penalty > 0
data_plane.enabled = true
data_plane.impl = transfer_queue
```

Start with the simple TQ backend to isolate model/training correctness, then
repeat with `mooncake_cpu`. The production claim requires the Mooncake run;
the simple backend is only a diagnostic precursor. Budget at least 500 GiB of
fresh Lustre space when optimizer state and a full checkpoint are retained.

After the Python-version P0 and image/venv probes pass, the source-owned entry
point is:

```text
/workspace/source/tools/qualification/production_nano_tq_fixed_batch.py
```

Run it with the image-owned driver interpreter, a scheduler-sanitized
environment, the exact source path above, and these required authorities:

```text
--config <source-owned Nano Megatron recipe>
--run-id <fresh unique ID>
--output-parent /workspace/output/production-nano-tq
--expected-source-root /workspace/source
--source-stack-id <sealed stack ID>
--source-bundle-manifest /workspace/source/SOURCE_BUNDLE.json
--source-bundle-manifest-sha256 <sha256>
--expected-model-root /workspace/model
--model-manifest /workspace/model/MODEL_MANIFEST.json
--model-manifest-sha256 <sha256>
--expected-image-digest sha256:<full-64-hex-digest>
--expected-image-fingerprint-sha256 <sha256>
--expected-driver-venv /opt/nemo_rl_venv
--expected-actor-venv /opt/ray_venvs/<qualified-mcore-venv>
--expected-python-version 3.13.13
--expected-ray-version 2.55.1
--expected-torch-version 2.11.0
--expected-cuda-compute-capability 9.0
--expected-num-nodes 1
--expected-gpus-per-node 8
--expected-world-size 8
--expected-tensor-parallel-size 8
--expected-pipeline-parallel-size 1
--expected-context-parallel-size 1
--expected-expert-parallel-size 8
--expected-train-global-batch-size 8
--expected-train-micro-batch-size 1
--expected-data-plane-backend mooncake_cpu
```

Hydra overrides must resolve the selected recipe to the fixed contract above,
including one training node, the local model snapshot, router replay disabled,
and the selected TQ backend. The harness records the full override vector and
the resolved semantic projection before Ray starts.

The dependency-free contract suite is intentionally runnable with an isolated
interpreter from a non-source working directory:

```text
NEMO_RL_EXPECTED_PRODUCTION_NANO_TQ_DRIVER_SHA256=<manifest file SHA> \
  /usr/bin/python3 -B -I \
  /workspace/source/tools/qualification/test_production_nano_tq_fixed_batch_spec.py
```

The test loads only the exact sibling harness path and checks the optional
scheduler-owned SHA. A pass that relies on the checkout being the current
working directory is not accepted as source identity evidence.

## Parallel diagnostic matrix

`production_nano_tq_diagnostic_matrix.py` defines four independent lanes:

1. exact actor-interpreter imports and CUDA/module provenance;
2. real TransferQueue PackedTensor round-trip with the `simple` backend;
3. the same round-trip with `mooncake_cpu`, fail-loud if Mooncake is absent;
4. the eight-H100 CP1 fixed-batch prev/ref/train/checkpoint lane above.

The matrix module plans and collects evidence; it intentionally does not submit
Slurm work. The scheduler-owned launcher starts each lane as a separate step or
allocation with an independent Ray namespace, port range, cache, output,
rank-log directory, Ray tmp directory, timeout, heartbeat, and GPU map. A lane
failure is collected after all lanes terminate and never cancels another lane.
The Mooncake API lane requires an exclusive hostname because the current
backend uses service ports 50050 and 50051. Running all lanes truly in parallel
needs four nodes and nine H100s; otherwise lanes may run serially without
changing their immutable plan identities.

The trusted host collector runs the full image/source/model verifiers exactly
once before the matrix and once after it. The resulting shared attestation binds
full content hashes, verifier records, bytes, device, inode, and mtime. Lanes
consume that immutable identity and check mountinfo plus module realpaths; they
must not independently rehash the 40+ GiB image or the entire model. Collection
fails if the before/after identity differs.

Every lane keeps stage/rank logs, Ray session logs, `PYTHONFAULTHANDLER=1`, and
bounded structured provenance. GPU lanes additionally use `NCCL_DEBUG=INFO`
and `TORCH_DISTRIBUTED_DEBUG=DETAIL`. The evidence allowlists environment names
and records credential variables as presence booleans only. Shell xtrace and a
full environment dump are forbidden.

The plan/collector contract test is also isolated from the checkout cwd:

```text
/usr/bin/python3 -B -I \
  /workspace/source/tools/qualification/\
test_production_nano_tq_diagnostic_matrix_spec.py
```

## Evidence and restart boundary

Every sample's PackedTensor media authority is joined to R3 fetch records from
all ranks for `prev_lp`, `ref_lp`, and `train`. The result additionally proves:

- finite current/reference logprobs on exactly the action-token mask;
- generation logprobs equal the just-computed current-policy logprobs;
- a finite, nonzero gradient on every required rank;
- a full-byte local trainable-parameter digest change after one optimizer
  update;
- one checkpoint tree joined to the controller result and every rank journal.
- successful TQ sample cleanup and worker shutdown before `RESULT.json` exists.

The rank journal has `baseline`, `optimizer-dispatched`,
`optimizer-applied`, and `checkpoint-joined` phases. It is deliberately
at-most-once: `optimizer-dispatched` without `optimizer-applied` is ambiguous
and the run must not be replayed. A durable restart claim still requires a
policy-owned distributed step journal and a fresh-policy checkpoint reload
whose parameters, optimizer, scheduler, and step identity match the completed
record. Until then, `restart_safe_replay=false` is release-blocking for
automatic recovery, though it does not prevent a one-shot qualification run.

## CP2 follow-up

CP2 is a separate two-node, 16-H100 lane (TP8 x CP2 x PP1, DP1). Do not relax
the CP1 harness in place. Add a successor harness that proves:

- every CP sibling receives the same decoded media digest before model-side
  context slicing;
- CP1 and CP2 agree on token/media masks and globally reduced eligible-token
  denominator;
- reference/current logprobs, loss, gradients, and parameter delta agree
  within an explicit numerical tolerance;
- checkpoint save and fresh reload work under the CP2 topology.

## Current qualification status

The source harness and dependency-free contract tests exist. They are not a
runtime qualification. A production-ready claim still requires:

1. resolve the Python 3.13.13 versus 3.13.14 contract;
2. attest all image/driver/actor dependency paths and exact versions;
3. pass the in-container full source-tree and HF snapshot rehash gates;
4. real H100 CP1 simple and Mooncake runs with no skipped gates;
5. fresh checkpoint reload and state/digest comparison;
6. the independent CP2 lane;
7. separate vLLM 0.20 and Gym f799 end-to-end OSWorld qualification.
