# OSWorld forest GRPO training (Nemotron 3 Nano Omni)

This branch holds the NeMo-RL source of an OSWorld GRPO run with forest training from captured rollouts and the
SingleController, plus its configs. Its files outside `3rdparty/` are byte-identical to the tree the run trained with (2,315 files compared).

## Source

| commit | content |
|---|---|
| `83385307` | NeMo-RL main (2026-10-17) |
| merge | #4200 head `c9c1b020` (checkpoint/resume for the SingleController), stacked on #4124 head `2f3649d75` (forest training from captured rollouts); conflicts resolved as listed in the merge message |
| test | the PR tests aligned with main (`fetch_prefix_chains`, forest attributes of a fake) |
| build(gym) | `3rdparty/Gym-workspace/Gym` → `1aa581c8` and the uv.lock built against it |
| fix(nemo-gym) | mask environments that fail before the first model call |
| fix(loss) | chunk the top-k/top-p logprob path in float32 (48k-token bins) |
| fix(data-plane) | broadcast host payloads in bounded staged chunks |
| fix(tq-policy) × 2 | logprob and training dispatch in rounds of at most `NRL_LOGPROB_MAX_SAMPLES_PER_RANK` samples per DP rank (host memory) |

The Gym commit `1aa581c8` (Gym main `b3103ea8` + #3591 + #4128 + #3368 + taskgen plugins and reward modes + served-call
history under token capture) is not on GitHub yet, so `git submodule update` cannot fetch it from the URL in `.gitmodules`.

## Configs

| config | what |
|---|---|
| `osworld-forest-smoke.yaml` | Smoke recipe on top of `vlm_grpo_nemotron_omni_clevr_vllm_token_capture_single_controller.yaml`: 2 × 4 per step, 15 actions, remote_docker VMs, 3 × 8 H100 (1 vLLM TP8 + 2 Megatron). |
| `osworld-forest-train.yaml` | First training run, on top of the smoke recipe: 4 × 8 per step, 200 actions, 7200 s per task, 48k packing bins, Megatron TP2 × CP8, lr 5e-6 constant, `all_captured` selection, checkpoint every step (4 h job chains). Reward `signed_v1` up to step 23, then `binary`: signed_v1 pays success with or without DONE, so the policy learned to run to the step cap instead of declaring DONE. |
| `osworld-forest-ae128.yaml` | Second run, on top of the smoke recipe: AgentEnv desktops through Gym's E2B provider (`osworld_agentenv`), 16 × 8 per step, 128 concurrent agents, 8 × 8 H100 (2 vLLM + 6 Megatron, DP3), binary reward, 25 steps. |

Environment the configs read:
- `OSWORLD_SERVING_ASSETS`: the chat template and the `nano_v3` reasoning-parser plugin.
- `OSWORLD_VM_PATH`: remote_docker runs only.
- `WANDB_ENTITY`.
- AgentEnv runs: `AGENTENV_URL`, `AGENTENV_API_KEY` and `AGENTENV_TEMPLATE`.
  - The agent's Python environment needs `e2b==2.46.0`, because the AgentEnv deployment serves `POST /sandboxes`.
  - A self-signed deployment also needs its certificate in certifi's bundle: `local_forwarder.py` uses requests with `trust_env=False`.
- remote_docker runs: `OSWORLD_REMOTE_*` for the VM host.

Compared with the configs as run, only the serving-asset paths, `vm_path` and the W&B entity (now environment interpolations) and some comment wording differ.
