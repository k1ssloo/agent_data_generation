# GEM Agent 数据合成项目：整体介绍与进度报告

更新时间：2026-08-16

## 1. 项目目标

本项目研究如何把 WikiHow 等流程文本转化为可训练的 Agent 数据。目标不是单纯生成更长的 tool-call 序列，而是构造满足以下条件的 semantic episode：

- Agent 只能看到用户目标、公开工具 schema 和运行时 observation；
- task、environment、solution 和 verifier 彼此一致，但 reference solution 与隐藏状态不泄露给策略；
- 后续动作真实消费早期 observation、handle 或中间产物；
- 环境状态能够触发分支、失败、诊断、修复、重试或替代方案；
- 所有关键参数都有可追踪的 argument provenance；
- 最终验证检查原始用户目标的状态谓词，而不是依赖一个名为 `verify_*` 的工具；
- 同一语义能力可以渲染成不同但可识别的 API，降低模型对固定工具名的依赖。

项目当前同时保留两条路线：

1. 原始 GEM 风格 Stage 1-4 pipeline，作为大规模 workflow SFT 基线；
2. 新的 task-first hidden-environment pipeline，作为长程、adaptive、可用于在线 RL 的主研究路线。

## 2. 当前代码结构

```text
gem_repro/
├── task_factory/       # WikiHow seed、task contract、bundle 与递归 operator
├── runtime/            # 隐藏状态、工具执行、episode runner
├── rollout/            # 模型在公开 policy context 中逐步 rollout
├── causal_validation/  # provenance、goal、counterfactual、ablation 等验证
├── training/           # rLLM SFT/RL adapter 与训练入口
├── schemas/            # task contract、bundle 等稳定 schema
├── prompts/            # 各生成和修复阶段的 prompt
├── scripts/            # 生成、materialize、验证、导出和批处理 CLI
├── tests/              # dependency-free 回归测试
├── data/               # 小型样例和可再生的 WikiHow 输入
└── outputs/            # 生成物；默认不进入 Git
```

`papers/` 位于工作区外层，用作研究参考，不属于可执行仓库，也不会随本次代码上传。

## 3. 数据合成流程

### 3.1 WikiHow seed 编译

原始流程文本先被编译成带原文证据的 seed。这里保留标题、步骤、source ID 和 source hash，区分原文事实与后续合成约束。相同 ID 如果对应不同文本，resume 会拒绝复用，避免 seed collision。

### 3.2 Task contract 生成

模型根据 seed 生成内部 task contract，包括：

- 用户可见目标；
- goal predicates 与 invariants；
- 环境中必须可发现的 evidence；
- 禁止的 shortcut；
- counterfactual axes；
- 期望的推理结构。

contract 是生成和验证阶段的内部真值，不进入 Agent 的 policy context。已通过结构验证的 contract 在正常 bundle repair 中保持不变，避免通过改写目标来“修复”失败样本。

### 3.3 Executable task bundle 生成

模型联合生成公开工具、隐藏初始状态、工具规则、reference plan 和 verifier 所需绑定。随后由确定性程序执行 oracle plan，检查 task、environment、solution 和 verifier 是否一致。

### 3.4 递归增难

递归生成以 parent bundle 为基础应用 transformation contract 和 patch，而不是重新生成整个任务。已实现的 operator 覆盖：

- target-specific audit evidence；
- environment-dependent execution route；
- asynchronous readiness；
- semantic failure、diagnosis、repair 与 retry；
- policy freshness 与 capacity reservation 等 fixture/operator 能力。

递归准入不只检查 `+steps`，还检查 parent plan 是否失效、是否增加可验证的语义复杂度，以及 counterfactual 环境是否需要不同策略。下一阶段的重点仍是增加 decision nodes、alternative recovery 和 decision entropy，而不是继续机械延长轨迹。

### 3.5 Hidden-environment rollout

Agent rollout 时只获得：

- instruction；
- public messages；
- public tool schemas；
- 每次实际调用产生的 observation。

隐藏初始状态、contract、reference plan 和 verifier predicates 均不对策略公开。模型必须在环境里发现能力、状态和失败原因，再组合工具完成任务。

### 3.6 确定性验证与导出

候选 episode 依次经过便宜到昂贵的 gate：

1. schema 与静态结构；
2. tool identifiability；
3. argument provenance；
4. state transition 与跨 observation 一致性；
5. oracle/reference execution；
6. goal evidence 与 final-state verification；
7. action ablation、counterfactual 和 adaptive topology；
8. SFT/RL export boundary 检查。

公开 SFT 只保留 OpenAI messages、公开工具和非敏感 lineage/metrics。RL task 只保留 instruction 与 bundle locator，训练时重新 reset 隐藏环境。

## 4. 已实现的核心能力

### 4.1 Grounding 与 provenance

validator 已从 handle provenance 扩展到 argument provenance。关键参数会分类为：

- `user_grounded`；
- `tool_observation_grounded`；
- `schema_grounded`；
- `agent_choice`；
- `derived`；
- `unexplained`。

敏感 literal 不能仅凭 schema enum 获得 grounding。未解释的 credential、recipient、revision、attachment 或内部 ID 会导致严格验证失败。

### 4.2 Tool identifiability 与 API rendering

alternate/opaque rendering 可以改变 function 和 parameter 名称，但必须保留公开、可区分的语义 description。静态 gate 会拒绝空 description、相同 schema 且不可区分的 tool group，避免把 lexical de-biasing 变成 tool identification 不可解。

### 4.3 因果验证

当前验证覆盖：

- delayed handle use 与 handle-chain depth；
- 中间对象和 evidence 的后续消费；
- semantic failure 是否被同目标的 diagnosis/repair 修复；
- async state transition；
- consequential action 的 prerequisite binding；
- 最后一次 mutation 后的只读 domain observation；
- goal predicate evidence；
- action ablation 与 necessary-action ratio；
- counterfactual 策略差异与 decision entropy。

固定的 `pending -> ready` 轮询只计 state transition，不会被误算为有熵的规划决策。

### 4.4 生成性能工程

当前 pipeline 已加入：

- parent + patch 的递归方式；
- deterministic renderer、metrics、exporter 和 validator；
- prompt/context slicing；
- exact-prompt LLM cache；
- root workers 与 bundle candidates 并发；
- resumable shards；
- guarded patch repair 与 full-repair fallback；
- candidate quality baseline 与 regression rejection；
- accepted semantic episodes/hour 等吞吐统计。

这些改动减少了重复生成和无效模型调用，但严格 vNext 批量数据的稳定 acceptance rate 仍需要在更大样本上实测。

## 5. SFT/RL 训练框架

当前选择 [rLLM](https://github.com/rllm-org/rllm) 作为 Agent SFT/RL orchestration layer，并使用其 veRL backend 做分布式 GPU 训练。原因是同一套 agent harness 可以覆盖：

- SFT；
- evaluation；
- GRPO；
- REINFORCE；
- on-policy distillation；
- 自定义 environment 和 evaluator。

仓库已经提供：

- `training/gem_environment.py`：可 reset 的隐藏因果环境；
- `training/rllm_adapter.py`：rLLM rollout/evaluator adapter；
- `training/train_rllm.py`：在线 RL 入口；
- `scripts/export_rllm_dataset.py`：SFT 与 RL task 双视图导出；
- `scripts/run_task_factory_batch.py`：生成、验证和导出的 resumable batch driver。

GPU 机器上的安装与训练命令见 `training/README.md`。当前 adapter 绑定到 README 中记录的 rLLM revision；升级依赖后应重新运行 contract tests。

## 6. 当前验证证据

本地已完成的验证包括：

- dependency-free test suite：112 passed，2 skipped；
- rLLM exporter/adapter 的本地 contract tests；
- 一个 strict vNext USB semantic episode 的完整导出；
- 该 episode 包含 23 个 tool calls、3 个 grounded observation-dependent decisions、1 次 semantic recovery、严格 provenance 和 final-state verification；
- 导出 manifest 确认 SFT 中没有 private fields，RL policy visibility 仅包含 instruction、public messages 与 public tools。

示例导出位于本地忽略目录 `outputs/training/rllm_vnext_usb_smoke/`，不会默认提交到 Git。可通过相同脚本重新生成。

## 7. 已知限制

1. 当前机器没有 GPU，因此尚未完成真实的 rLLM/veRL SFT 或 GRPO 训练，也没有训练后 benchmark 结果。
2. 完整 `rllm[verl]` 分布式依赖没有在本机安装；当前验证覆盖 adapter contract 和数据边界，不等价于 GPU 集群端到端训练成功。
3. strict vNext 的大规模 cold-start batch 尚未完成，因此其每条 accepted episode 的真实平均时延、acceptance rate 和成本仍待测量。
4. 旧的 3261 条 Stage 1-4 corpus 适合作为通用 workflow/tool SFT baseline，但不能全部视为高 decision-density 的 adaptive trajectory。
5. 现有 strict 样本证明了垂直链路可行，但样本量还不足以支撑“长程 adaptive 数据已经规模化”的结论。
6. 通用 alternative-plan search、更多 failure topology、跨 domain state consistency 和模型 rollout 后的独立语义审计仍需继续扩展。

## 8. GPU 机器上的建议执行顺序

```bash
# 1. 回归测试
python3 -m unittest discover -s tests -p 'test_*.py' -v

# 2. 小批量生成并验证
python3 scripts/run_task_factory_batch.py \
  --input data/wikihow_computer_10000.jsonl \
  --output-dir outputs/task_first/batch_adaptive_v1 \
  --config ../config.toml \
  --provider responses \
  --limit 30 \
  --shard-size 10 \
  --workers 4 \
  --bundle-candidates 2 \
  --recursive-generations 2 \
  --resume \
  --continue-on-error

# 3. 导出 SFT 与在线 RL task
python3 scripts/export_rllm_dataset.py \
  --validation outputs/task_first/batch_adaptive_v1/validation.jsonl \
  --output-dir outputs/task_first/batch_adaptive_v1/training

# 4. 安装训练框架；固定 revision 见 training/README.md
uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git@9beb6e0f676a46d38858991fd79ac5f8e0b16d4c"

# 5. 先做小规模 SFT smoke，再扩大数据/卡数
rllm sft \
  --train-file outputs/task_first/batch_adaptive_v1/training/sft.jsonl \
  --model Qwen/Qwen3.5-4B \
  --backend verl \
  --gpus 8 \
  --max-length 32768 \
  --tokenize-method hf_template
```

建议先用 30 条样本完成“生成 -> 严格验证 -> export -> tokenize -> 单次训练 step”的端到端 smoke，再扩展到 1,000 条以上。生成和训练应分别记录 wall time、模型调用数、acceptance rate、accepted episodes/hour、GPU utilization 和训练 loss。

## 9. Git 与数据安全边界

本仓库只上传源码、prompt、schema、小型 fixture、测试和说明文档。以下内容不应进入 Git：

- 工作区根目录的 `config.toml`，其中包含模型 provider 配置和 API credential；
- `.env`、PAT、API key、private key；
- `outputs/` 下的模型响应、缓存和大规模生成物；
- 本机 Codex state、SQLite 日志、memory/goal 文件；
- 论文 PDF、模型权重、虚拟环境和下载缓存；
- 未经确认可公开的原始或衍生数据集。

需要迁移大型训练数据时，应使用对象存储、Hugging Face Dataset 或受控的 Git LFS 仓库，并与代码仓库分开管理。

## 10. 下一阶段里程碑

1. 在 GPU 机器完成 rLLM/veRL import、tokenization、SFT 和在线 rollout smoke；
2. 用 30 个 WikiHow roots 测量 cold-start latency、acceptance rate 和 strict/adaptive tier 吞吐；
3. 将 recursive operator 的优化目标从 `+steps` 进一步改为 `+decision nodes`；
4. 每条强样本目标为 15-25 个 meaningful calls、3-5 个 observation-dependent decisions、1-2 次 semantic failures 和至少一条真正 alternative recovery path；
5. 扩展跨 observation temporal consistency 与 instruction-to-goal coverage；
6. 分层构建 linear、branching、recovery 和极长 episode mixture，再开展 SFT/RL ablation；
7. 在未见 tool names、未见 schema rendering 和 benchmark tools 上评估组合迁移。

当前最准确的项目定位是：**task-first、hidden-environment、causally validated Agent data factory 已完成可运行的垂直切片和训练适配，但严格 adaptive 数据的规模化生产与 GPU 训练实验仍处于下一阶段。**
