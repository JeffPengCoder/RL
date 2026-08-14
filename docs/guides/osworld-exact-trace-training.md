# OSWorld exact-trace 全轨迹训练

## 0. 当前状态与边界

本集成分支：

```text
NeMo-RL primary parent: NVIDIA-NeMo/RL PR #3642@022269324326b8b2680dca8d7bd7bd78d78d2998
NeMo-RL capability parent: JeffPengCoder/RL feature/gym-osworld@3095f8519507d7094fc3a840b664143f3c5ca4d2
NeMo-RL combined branch: JeffPengCoder/RL 3642+
base: aroshanghias/context-compaction-v2-clean@42a65427dce038f57f7fd8eed6a24f6a8ce72c2b
one-step qualified NeMo-RL: 4ce0a05961ad7b90a4ef669ce3da37c87ce3bb41
dense-modality qualified NeMo-RL: c1d7b2fc9403fa1b21dfc75dfdd3d70fcd4da1b9
Gym: feature/osworld-exact-trace-training@e191ef90b5175be57f142da451186a14ec530e4a
Rohit comparison: rohit/gymv-mm-integration@71717873240c99fd1ede2db17480818205766848
```

它把 Gym 的通用 OSWorld trajectory 合同接到 Arash 已实现的 training-side token
stream reconstruction。Gym 不再提供 `training_mode`、`last`、`all` 或
`exact_trace` strategy：benchmark 和 training 执行完全相同的 agent prompt/action
逻辑，差别只在下游 consumer 是否具备训练准入条件。

父分支的合同、重建、校验、配置和单元测试已经具备，并完成过以下真实
OSWorld VM/GPU gate；这些结果是合并输入证据，不等于新的 `3642+` merge SHA 已验收：

```text
generation-only trace qualification: passed, Job 15436343
1-step optimizer/checkpoint smoke: passed, Job 15479566
16-step fixed-validation run: not run on this exact-trace branch
checkpoint/resume qualification: not run
3642+ combined SHA generation/optimizer gates: pending
```

Job 15479566 的 8 条 rollout reward 全为 0，loss 为 0；它证明 logprob、backward、
optimizer 和 distributed checkpoint 控制路径可执行，不证明非零梯度或模型提升。
因此可以说 one-step 执行链路已通过，不能说完整训练效果或生产长跑已经验收。

## 1. 最初命中的问题：token 2,080 开始前缀不一致

在 Rohit `71717873` 基线上，把 OSWorld 多轮输出设置为 `all` 后，第二次 model call
在 token index 2,080 开始与上一轮累计 token stream 不同，随后触发：

```text
Non-contiguous messages found
```

这不是一个可删掉的多余 assert，而是表示 trainer 将要错误组装样本。

普通 append-only 多轮对话可以写成：

```text
turn 1: P1 | G1
turn 2: P1 | G1 | delta-P2 | G2
turn 3: P1 | G1 | delta-P2 | G2 | delta-P3 | G3
```

其中 `P` 是 prompt tokens，`G` 是 assistant generation tokens。Rohit 的实现维护
一条全局 `seen_token_ids`，要求下一轮 prompt 以此前的 `prompt + generation` 为
严格前缀。

OSWorld/Nano Omni 并不满足这个假设。每执行一次代码后，桌面状态发生变化；agent
收到新截图，并可能重新组织历史：

```text
P1 = system + task + screenshot-1 + chat_template
G1 = first sampled action/code

execute G1; desktop changes

P2 = re-rendered system
   + selected/reformatted action history
   + current bounded screenshot window
   + chat_template applied again
G2 = second sampled action/code
```

`P2` 可以与 `P1 | G1` 没有 token 前缀关系。即使只讨论 assistant tokens，问题仍然
存在：`G2` 是 rollout server 在真实 `P2` 下采样的；如果 trainer 强行构造一个假的
`P1 | G1 | ... | G2`，它为 gradient 重新前向计算 current/reference logprobs 时，
是在错误上下文下给 `G2` 算概率。被训练的 generation 仍是 server 采样的 `G2`，
错误发生在 NeMo-RL 为 loss 准备的前向条件上。

## 2. 抽象：trajectory 不是 trainable token representation

Gym 拥有环境语义。第 `t` 步可以记录为：

```text
transition_t = (
  state_t,
  action_t,
  reward_t,
  done_t
)
```

OSWorld 的 `state_t` 包含真实 prompt view 和有序图片；`action_t` 包含原始 model
completion 与解析后的桌面动作。这个 trajectory 适合环境重放、奖励归属、debug 和
审计，但还不是 trainer 可以直接送进模型的 tensor row。

本分支使用三层表示：

```text
Gym environment transitions
  desktop state/action/step reward/next state/done
        |
        | group 1..N policy calls into each real desktop transition
        v
Gym trajectory_model_calls
  exact materialized prompt messages + ordered media references
  raw completion + parsed action + parser result
  reward/done/eligibility linkage
  optional token IDs/logprobs
        |
        | when the model server exposes complete exact evidence
        v
authoritative model-call evidence
  prompt_token_ids
  sampled_token_ids
  sampled_logprobs
  ordered media_ids
  rollout/completion/action/request identities
  policy-output spans and eligibility
        |
        | validate token/media continuity
        v
NeMo-RL trainable representation
  one logical rollout
  -> one or more prefix-contiguous physical traces
  -> prompt tokens mask=0
  -> eligible sampled action tokens mask=1
  -> multimodal tensors in exact media occurrence order
```

`trajectory_transitions` 与 `trajectory_model_calls` 对任何 model endpoint 都存在；
后者才是“trajectory 还不是最终 tensor row”中间那层，保存 trainer 重建所需的真实
prompt/action 语义。图片按内容寻址存入一次 `media_assets`，每轮 prompt 只存有序
`media_id`，避免重复 base64。

NeMo-RL 会核对 trajectory identity、transition/model-call 归属、prompt media 顺序、
reward/done/eligibility，以及 exact `completion_evidence` 的 tokens/logprobs。重复的
token/logprob 数组在校验后从训练日志投影中移除；训练权威是 trace bundle 与 physical
message logs，语义记录用于审计和 debug。

## 3. Logical rollout 与 physical trace

对相邻两次 model call，NeMo-RL 独立判断：

```text
token_contiguous
  = previous prompt+generation is a prefix of current prompt

media_contiguous
  = previous ordered media IDs are a prefix of current ordered media IDs

append_compatible
  = token_contiguous AND media_contiguous
```

若不连续，就开始新的 physical trace。三轮都重写 prompt 时：

```text
logical rollout R
reward = reward(R)
advantage = advantage(R)

physical trace 0: P1 [mask=0] | G1 [mask=1]
physical trace 1: P2 [mask=0] | G2 [mask=1]
physical trace 2: P3 [mask=0] | G3 [mask=1]
```

同一 logical rollout 的所有 eligible generation 都共享该 rollout 的 reward 和
advantage，但 physical trace 数量不能被当作额外 reward 样本。GRPO comparison
group、optimizer boundary、scheduler progress 和 loss normalization 仍按 logical
rollout 语义处理。

如果两轮确实保持 token 和 media 前缀连续，重建器可把它们保留在同一 physical
trace 中，只记录新增部分。

## 4. 当前方案、旧方案与训练准入

旧 Rohit 方案曾提供 `last/all`：`last` 只训练末轮，`all` 暗含 append-only prompt。
它们用于解释历史实验，不是本分支的可选执行路径。

当前 Gym 总是执行同一条链：

```text
任何 benchmark / model test
  -> trajectory_contract
  -> trajectory_transitions
  -> trajectory_model_calls

若 endpoint 同时返回完整 prompt tokens、sampled tokens、sampled logprobs
  -> 自动附加 exact model-call authority

若 NeMo-RL training manifest 还提供 trajectory_identity
  -> 自动选择 trace-aware trainer
  -> 绑定 launcher-owned runtime contract
  -> 满足全部条件才进入 loss
```

因此，benchmark 不需要 contract-v2 开关，也不会因为模型不返回 logprob 而失败；它
仍有完整语义轨迹。训练 manifest 的 `trajectory_identity` 不是让 Gym 改 prompt 的
mode switch，而是 logical rollout/group/retry 的 caller-owned identity。缺 exact evidence
或 runtime admission 时，trainer fail closed，不能退回末轮或错误拼接。

Arash 基线仍保留名为 `context_compaction_training` 的内部实现结构，供已有 scripted
compaction recipe 兼容；OSWorld recipe 不再要求用户设置这个开关，NeMo-RL 从
`trajectory_identity` 自动选择该 physical-trace 数据路径。

### 4.1 Training、训练期 eval 与独立 benchmark

三条执行链不能混为一条：

```text
standalone benchmark
  external vLLM -> Gym benchmark runner -> OSWorld
  不启动 NeMo-RL；请求不带 rollout_purpose

RL training
  NeMo-RL generation_only=false -> Gym -> OSWorld
  NeMo-RL 在通用 BaseRunRequest 写 rollout_purpose=training
  exact trace 进入 loss/backward

training-time evaluation
  NeMo-RL generation_only=true -> Gym -> OSWorld
  NeMo-RL 写 rollout_purpose=evaluation
  只统计 held-out 曲线，不进入 loss/backward
```

`rollout_purpose` 是 scheduler-owned 的通用请求语义，不是 OSWorld 的训练模式开关；
所有 Gym agent 都能接收它，默认可以忽略。OSWorld 需要消费它，是因为 parser retry
会产生额外 policy call：训练 profile 使用 `parse_retries=1`，训练期 evaluation 使用
`parse_retries=5`。NeMo-RL 内部 vLLM 仍严格校验请求：training 必须匹配 on-policy
`1.0/1.0/768`；只有 scheduler 标记的 evaluation 才能使用单独固定的
`0.6/0.95/4096`，不能用 purpose 绕开任意采样参数检查。

由 NeMo-RL 写这个字段是合理的，因为训练 scheduler 才知道某批 rollout 属于 optimizer
采样还是训练期 validation；Gym 和 model server 不应从 payload 形状、是否请求 logprob
或 endpoint 名称猜测。但跨服务合同不应该是模糊的 `is_eval` boolean，而应是可校验的
`rollout_purpose=training|evaluation`。是否参与 loss 仍由 NeMo-RL 本地的
`generation_only`/训练控制流决定，不能由 Gym 反向决定。换句话说，这是两个概念：

```text
rollout_purpose
  = 本次调用应采用哪一个显式、固定的生成/解析 profile

trainable / generation_only
  = 返回结果是否允许进入 trainer 的 loss/backward
```

正常训练和训练期 validation 中两者分别一一对应；若未来增加“采用 training sampling
但只收集 trace、不做 backward”的 preflight，应在 NeMo-RL API 中显式解耦这两个轴，
而不是让 Gym 猜测或再增加一个 OSWorld training-mode 开关。独立 benchmark 没有
NeMo-RL scheduler，因此不要求携带该字段，继续使用 Gym 的 standalone 默认合同。

Nano Omni 当前 recipe 固定 `chat_template_content_format=string`。实测使用 `openai`
block-list 格式会让历史 assistant content 在第二轮被模板渲染成 Python list literal，
模型随后输出同类 list 文本并触发 parser failure。旧的约 50% standalone baseline 同样
使用 string content format。

独立 benchmark 使用当前 Gym 外部驱动：先由 `gym env start`（或 OSWorld 的
`benchmarks/osworld/tools/start_control.sh`）启动环境/agent control plane，再运行
`gym eval run --no-serve`（或 `run_eval.sh`）收集任务。旧的 `ng_run`、
`ng_collect_rollouts` 和 `run_omni_mini_vllm.sh` 只属于历史复现记录，不能作为新入口。
不要为了复用 NeMo-RL 的资源启动逻辑而把 benchmark 塞进 one-step GRPO driver。
训练期 evaluation 可以用较小 held-out 集和较少重复数看趋势；若它的 `max_steps`、
任务集或 sample 数与正式 benchmark 不同，必须明确标记为 validation curve，不能与
正式 benchmark 分数直接横比。

## 5. 本集成分支增加的 NeMo-RL 行为

除 Arash 基线已有的 logical-to-physical 重建外，本分支加入：

1. Gym submodule 固定到可访问的 `JeffPengCoder/Gym`，branch metadata 为
   `feature/osworld-exact-trace-training`，gitlink 是唯一版本权威。
2. 通用 `trajectory_transitions` + `trajectory_model_calls` 逐项校验；语义 trajectory
   与 exact generation evidence 不一致时，在 loss 前 fail closed。
3. OSWorld exact-trace 16-step recipe 与 OpenSandbox overlay。
4. 固定、无泄漏的 train/validation split 工具，并自动补通用 `trajectory_identity`。
5. `HF_MODULES_CACHE` 优先级与 PYTHONPATH 去重，避免完整动态模块位于 writable cache
   时 Ray actor 误绑定不完整 `HF_HOME/modules` seed。
6. pooled Ray initializer 继承 PYTHONPATH/venv；构造参数包含 trust-remote-code 类时也能
   在反序列化前找到动态模块。
7. `training_node_resource` / `inference_node_resource` role affinity，避免异构节点按
   Ray join order 放反训练和 vLLM。
8. `NRL_REFIT_BUFFER_SIZE_BYTES`：异构 GPU collective 使用同一 packed-buffer 大小。
9. 自动 HF -> Megatron conversion 只有在完整 metadata 文件存在时才算 cache hit，
   避免把半截 distcp 目录当成成功 checkpoint。
10. vLLM response 中出现 NaN/Inf 时报告准确 JSON path，而不是在下游得到模糊错误。
11. scheduler-owned `rollout_purpose`、独立 evaluation sampling contract，以及
    OSWorld training/evaluation parser-retry profile；standalone benchmark 保持外部入口。

没有纳入 feature branch 的内容包括开发者本机 `.dockerignore` 偏好、临时 source
overlay、秘密文件和只适用于某个旧 image digest 的 venv 版本检查。这些属于部署
artifact 或本地状态，不是共享仓库 invariant。

## 6. 运行前提：不要假设新用户已经具备

至少需要：

- 可读取本 NeMo-RL fork/branch 与 Gym submodule 的 GitHub 凭证；
- cw-dfw SSH、Slurm account/QoS/partition、Pyxis/Enroot 和两台 8-GPU AMD64 节点；
- 或等价的两节点 Ray 集群，并能分别注册 `nrl_trainer_node=1` 与
  `nrl_vllm_node=1`；
- 两节点都可读写的 Lustre 目录，容量足够存模型、约 58-GB `.sqsh`、checkpoint、
  rollout artifacts、uv/HF/Transformers/Megatron/vLLM/torch compiler caches；
- 完整 Nano Omni snapshot。`preprocessor_config.json` 引用的 `processing.py` 和其他
  trust-remote-code 模块必须真实存在，不能只复制 weights/config symlink；
- OSWorld task JSONL 与任务所需的下载资产/网络权限；
- OpenSandbox API base URL、API key、`osworld-kvm` pool 权限，或本地/远端 KVM qcow2
  与 Docker Sandbox；
- 对 proxy-required task，需要 OSWorld 结构化 proxy JSON；普通 shell `HTTP_PROXY`
  不能代替它；
- W&B 账号/凭证仅在启用 W&B 时需要，secret 只通过环境或私密文件传入。

Colossus bare-metal reservation和 cw-dfw Slurm 是两个控制面；有其中一个权限不代表
自动拥有另一个。长期作业还需要在 lease/time limit 前保存 checkpoint、缓存和
successor capacity。

## 7. Checkout 与 submodule

```bash
git clone --branch '3642+' --recurse-submodules \
  https://github.com/JeffPengCoder/RL.git nemo-rl
cd nemo-rl
git submodule sync --recursive
git submodule update --init --recursive

git rev-parse HEAD
git -C 3rdparty/Gym-workspace/Gym rev-parse HEAD
```

第二个 SHA 必须与第一个 commit 中记录的 gitlink 完全一致。不要在运行节点只手改
submodule 而不记录 gitlink。

## 8. 准备固定训练集与验证集

例如只训练/验证 Chrome，固定选 8 个 held-out validation tasks：

```bash
python examples/nemo_gym/prepare_osworld_exact_trace_data.py \
  --input 3rdparty/Gym-workspace/Gym/benchmarks/osworld/data/example.jsonl \
  --domain chrome \
  --validation-count 8 \
  --seed osworld-chrome-r1 \
  --train-output /lustre/.../data/osworld-chrome-train.jsonl \
  --validation-output /lustre/.../data/osworld-chrome-validation.jsonl
```

示例文件若筛选后任务数不足 9 条会正确失败；正式运行应使用完整 OSWorld manifest。
split 按 `seed + task_id` 的 SHA-256 排名，因此输入行顺序变化不改变 held-out task
集合。train 和 validation task ID 必须不相交；验证集固定后不能按曲线结果反复挑选。

工具为每行添加通用 caller-owned identity：

```text
trajectory_identity.schema_version = 1
trajectory_identity.group_id
trajectory_identity.task_id
trajectory_identity.rollout_index = 0
trajectory_identity.attempt_index = 0
agent_ref = osworld_simple_agent
```

GRPO 在每个 prompt 被 repeat 成多次 generation 后，为 replicas 分配不同 rollout
index。attempt index 只表达同一个 logical rollout 的 retry，不能用于伪装新的采样。

## 9. 环境变量与持久化目录

```bash
export NANO_OMNI_MODEL_NAME=/lustre/.../snapshots/<immutable-revision>
export OSWORLD_TRAIN_DATA=/lustre/.../data/osworld-chrome-train.jsonl
export OSWORLD_VALIDATION_DATA=/lustre/.../data/osworld-chrome-validation.jsonl
export OSWORLD_CHECKPOINT_DIR=/lustre/.../runs/osworld-exact-trace/checkpoints
export OSWORLD_LOG_DIR=/lustre/.../runs/osworld-exact-trace/logs

export HF_HOME=/lustre/.../cache/huggingface
export HF_MODULES_CACHE=/lustre/.../cache/huggingface/modules
export NEMO_RL_VENV_DIR=/lustre/.../cache/nemo-rl-venvs
export NEMO_GYM_VENV_DIR=/lustre/.../cache/nemo-gym-venvs
export UV_CACHE_DIR_OVERRIDE=/lustre/.../cache/uv
export NRL_REFIT_BUFFER_SIZE_BYTES=536870912

export OPENSANDBOX_BASE_URL=<provided-service-url>
export OPENSANDBOX_API_KEY=<secret>
export OPENSANDBOX_POOL_REF=osworld-kvm
```

在启动 Ray/NeMo-RL 之前，必须先用同一份 Gym 配置预热
`NEMO_GYM_VENV_DIR`，再显式安装 OSWorld 没有随 Gym 默认环境分发的桌面运行时
依赖：

```bash
gym env prefetch
bash 3rdparty/Gym-workspace/Gym/responses_api_agents/osworld_agent/install_optional_runtime_deps.sh \
  "${NEMO_GYM_VENV_DIR}/responses_api_agents/osworld_agent/.venv"
```

recipe 的 `skip_venv_if_present: true` 只复用已经存在且可执行的 venv；首次运行
仍会创建 venv。生产 launcher 还应在启动 trainer 前执行 Gym 的
`runtime_dependencies.py check`，不能把仅有 `bin/python` 当作依赖完整的证据。

`NRL_REFIT_BUFFER_SIZE_BYTES` 必须在 collective 的所有 rank 上完全相同。cache 可加速
重跑，但 cache 不是正确性证据；checkpoint、result/trajectory、resolved config、
source SHA 和完成 marker 才是。

## 10. 三阶段运行

### 10.1 Generation-only preflight

```bash
uv run --locked --extra mcore --extra vllm \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_opensandbox_exact_trace.yaml \
  env.nemo_gym.is_trajectory_collection=true \
  checkpointing.enabled=false
```

检查 `trajectory_collection.jsonl`：

```text
logical rollout/model-call count正确
所有 sampled token/logprob 数量对齐
P2 非前缀时存在 boundary，且产生新的 physical trace
media ID 顺序与每轮截图窗口一致
trajectory transition/model-call identity 与 completion evidence 一致
每个 materialized prompt、raw completion、parser retry 都可审计
基础设施失败标记为 mask_sample，不伪造成 reward=0
```

### 10.2 One-step optimizer smoke

```bash
uv run --locked --extra mcore --extra vllm \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_opensandbox_exact_trace.yaml \
  grpo.max_num_steps=1 \
  checkpointing.save_period=1
```

必须看到 8/8 logical rollouts、current/reference logprob、optimizer、refit、完整
checkpoint 和正常 driver exit；仅看到 VM 或 vLLM 启动不算训练成功。

已归档的真实 gate 是 Job 15479566：8/8 rollout、current logprobs、`run_backward`、
65.26 秒 policy training 和 distributed optimizer-state save 全部完成，driver/parent
为 `COMPLETED 0:0`。最终 `step_1` 有 18 个文件、440,346,575,025 字节。由于 8 条
reward 全为 0 且 loss 为 0，这个结果不覆盖“非零 learning signal”验收。

### 10.3 16-step curve

```bash
uv run --locked --extra mcore --extra vllm \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_osworld_opensandbox_exact_trace.yaml \
  logger.wandb_enabled=true \
  logger.wandb.entity=<entity> \
  logger.wandb.project=<project> \
  logger.wandb.name=osworld-exact-trace-16step
```

默认在 step 0/4/8/12/16 对同一 held-out set 做 pass@1 风格评测，并在 step
4/8/12/16 保存 checkpoint。训练 reward 曲线不能代替固定验证曲线。

当前 embedded vLLM HTTP gate 要求请求 sampling 与 server generation config 完全一致，
所以训练进程内 validation 默认仍用训练 temperature=1.0。若要比较 temperature=0.1
或 0.6，应对同一固定验证集启动独立 eval-only replay，并把 checkpoint、task IDs、
temperature 与样本数一起记录；不能在同一次训练中悄悄改变 sampling contract。

### 10.4 当前 qualification ledger

```text
Job 15436343
  exact generation-only
  435/435 sampled/trainable tokens aligned

Job 15479566
  exact one-step training
  8/8 rollouts; logprobs/backward/optimizer/checkpoint completed
  driver/parent 0:0; step_1 = 18 files / 440,346,575,025 bytes
  all rewards and loss are zero; no policy-improvement claim

Job 15482734
  accepted-image dense-modality batch regression gate
  three batch orderings passed; parent/batch/step 0:0
  accepted follow-up commit c1d7b2fc9403fa1b21dfc75dfdd3d70fcd4da1b9

Gym 54056bba8
  evaluator postconfig without an explicit return-code policy preserves
  upstream OSWorld best-effort behavior; a missing agent-created artifact is
  scored as reward=0 instead of being mislabeled as an infrastructure mask
  73 OSWorld client tests passed locally
```

完整 369-task benchmark 固定使用 one-step gate 对应的 immutable NeMo-RL 4ce0 和
Gym 1b0，避免中途换源码；它必须在 369 attempted 与 361 no-GDrive 两个视图都完成、
最后一个 Slurm attempt 已终止且 anomaly sets 为空以后再报告。该 benchmark 不是
optimizer gate，也不能替代尚未执行的 16-step exact-trace fixed-validation curve。
Gym 54056bba8 是 benchmark 启动后根据 raw artifacts 定位出的后继修复；它不会被
热切换进这次固定源码的 benchmark，也不会改写原始 `result.json`。最终报告必须另列
official-compatible 分类，并用后继源码对命中的 evaluator 样本做 focused rerun。

## 11. OpenSandbox 的任务、VM 与 proxy 生命周期

在 OSWorld agent 中，一次 Gym `/run` 是一个完整 task rollout，不是单个 desktop
action。OpenSandbox VM 在该 `/run` 建立 `DesktopEnv` 前创建/取得，完成 guest readiness
gate 后执行 task reset、最多若干次 model-call/action、inline evaluator，最后释放
sandbox。一个 task 内的多轮 action 使用同一 VM。

OpenSandbox connection 通过 `OPENSANDBOX_BASE_URL` 和 API key 建立；Pool 的 gateway
endpoint 由 provider 返回，Gym adapter 建本地 forwarder，并把可达的 endpoint 交给
OSWorld controller。对于 task metadata 中 `proxy=true` 的任务，另需结构化 OSWorld
proxy 配置；是否允许无 proxy 直连由 agent config 明确控制。API key/proxy secret 不应
写入 YAML、trajectory 或 git。

## 12. mask、reward=0 与训练资格

```text
reward=0 + eligible=true
  = agent真实执行但任务失败，可作为负样本

mask_sample=true / eligible=false
  = VM、网络、timeout、evaluator或证据不完整，不能伪装成策略失败
```

Arash 基线当前对 CC training 中的 masked/truncated logical rollout fail closed。长期
OSWorld 训练理想上需要在 advantage 之前按整个 comparison group 重采样，直到组完整；
不能先展开 physical rows 再只删除坏掉的一段。本分支尚未声称已经实现该 group-level
resampling，这是首轮真实 qualification 的已知限制。

## 13. Rohit 与 Arash 如何合并

两条线的 merge base 是 `8b5c6de5263de72e6229ab00ac51e774ba83a519`。相对该
base，Rohit 有 18 个独有提交，Arash 有 110 个独有提交；不要直接把两条长期分支
blind merge。

推荐：

```text
以 Arash/newer-main 为 trainer 基线
  -> 使用本分支固定的 Gym exact-trace submodule
  -> 对 Rohit commits 做 capability audit
  -> 只移植新基线仍缺的行为与经过验证的 recipe 参数
  -> 每个行为单独测试
```

Rohit 的早期 Nano Omni/vLLM/Megatron/async 实现多数已由更新主线以不同代码取代；
不要按 commit hash 整包 cherry-pick。chat-template kwargs、sequence length、generation
count 等属于显式 recipe 决策，应手工合并并记录 resolved config。旧 `last` 行为只
适合复盘旧实验，不能替代 exact-trace trainer architecture，也不在本分支的 OSWorld
执行链中。

审计命令：

```bash
BASE=8b5c6de5263de72e6229ab00ac51e774ba83a519
git log --oneline ${BASE}..rohit/gymv-mm-integration
git log --oneline ${BASE}..aroshanghias/context-compaction-v2-clean
git range-diff ${BASE}..rohit/gymv-mm-integration \
  ${BASE}..aroshanghias/context-compaction-v2-clean
```

## 14. 最终验收证据

至少保存：

```text
NeMo-RL exact SHA and dirty-state proof
Gym gitlink exact SHA
container digest / sqsh SHA and size
model snapshot revision and dynamic-module completeness report
resolved Hydra config and non-secret environment manifest
train/validation task-ID split manifest
trajectory exact-trace integrity report
W&B URL plus local metrics export
checkpoints with all rank shards and checksums
both-node compiler/cache snapshot checksums
driver exit, diagnostics complete, result counts
```

“生成了 `.sqsh`”“Pyxis import 完成”“vLLM ready”或“出现 step 行”都只是中间状态，
不能单独证明 full resolver、container gate、push 或训练成功。
