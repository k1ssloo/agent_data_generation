# Task-First 可验证长程数据重构计划

## 1. 文档目的

本文档给出 `gem_repro` 从当前 GEM 风格的“过程文本 -> workflow/tool bank ->
one-shot trajectory -> replay validation”流程，迁移到“可执行任务生成 -> oracle
可解性证明 -> 隐藏环境 rollout -> outcome/causal validation”流程的完整实施计划。

方案吸收《Recursive Synthesis for Long-Horizon Terminal Tasks》（RST）的核心原则：

- 合成单位是完整、可执行、可验证的任务，而不是预先写好的轨迹；
- reference solution 只用于证明任务可解，不能作为 agent 的公开执行路线；
- verifier 检查任务结果和语义约束，而不是要求复现固定动作序列；
- public instruction、observable environment、reference solution 和 private verifier
  必须共同满足一致性约束；
- 已验证任务可以作为后续递归扩展的 seed。

在此基础上，本项目还需要补充 RST 没有直接解决的问题：有效 branching
factor、替代解、多状态干预、观察驱动决策，以及对未见工具接口的迁移能力。

本文档是实施计划，不直接修改现有数据 schema、prompt 或生成结果。

## 2. 当前方向的问题

### 2.1 轨迹优先导致策略泄漏

当前 Stage 2 直接生成 `workflow.execution_graph`，Stage 3 同时接收 workflow、
完整 environment 和 tool rules。模型的主要工作因而接近于沿已知图补全调用，而不是在
未知状态中发现 affordance、诊断问题并形成计划。

### 2.2 可回放不等于有推理性

`text-exec-dsl-v0` 可以证明 tool response 与合成状态一致，但不能证明：

- 某个调用对完成任务是必要的；
- 后续动作确实因前序观察而改变；
- 失败后的动作解决了相同错误；
- 最终验证覆盖原始任务目标；
- 环境存在不止一条窄的 oracle 路径。

### 2.3 固定 tool bank 容易产生接口记忆

canonical tool bank 降低了任务专用 API 的比例，但也使训练数据反复出现固定名称、
参数和调用搭配。小模型可能学习 `list_records -> get_record -> update_record` 等表面模板，
而不能把相同能力迁移到 benchmark 中不同的工具 schema。

### 2.4 长度指标会奖励机械工作

tool-call 数、command 数、solution 行数只反映工作量，不直接等价于推理难度。重复读取、
重复 CRUD、固定轮询和批量文件操作都可以拉长轨迹，却不增加观察驱动的决策深度。

### 2.5 模型同时生成环境和答案会形成自洽闭环

同一类模型若同时决定 task、environment、oracle trajectory 和 validation target，容易生成
一个对自身答案友好的封闭世界。即使所有组件内部一致，也可能存在 oracle 信息过多、
branching factor 过低、shortcut 未被 verifier 拒绝等问题。

## 3. 重构目标

### 3.1 核心目标

新 pipeline 必须保持“模型生成 + 因果可验证性”：

- 模型用于提出自然任务扩展、构造公开契约、生成 reference solution、生成 verifier
  候选、生成环境变体和执行 rollout；
- 确定性程序负责 schema/type 检查、sandbox 执行、状态记录、契约对齐、反捷径、
  干预实验、目标验证和最终数据准入；
- 任何模型自报的 `long_horizon`、`causal` 或 `verified` 标签都不能直接作为质量证据。

### 3.2 任务质量目标

一条高质量长程任务应自然包含若干结构：

- 早期发现的 handle 在多个中间动作后才被消费；
- 早期读取的权限、额度、配置或策略决定后续分支；
- 中间操作生成的新对象成为后续调用输入；
- 异步任务状态决定等待、重试、取消或回滚；
- 错误恢复针对明确 error class，并改变相关状态或参数；
- 最终验证检查原始 goal predicates，而不是仅调用名称为 `verify_*` 的工具；
- 至少存在替代计划，或在合理状态干预下产生不同的正确计划。

### 3.3 工具泛化目标

模型应根据当前 episode 的工具描述、schema、帮助信息和执行反馈推断工具能力，而不是依赖
训练期间稳定的工具名。训练和评测需要区分以下泛化层次：

1. 未见工具名称；
2. 未见参数和返回 schema；
3. capability 的 split/merge；
4. 未见能力组合；
5. 完全未见能力。

前三类是本次重构的直接目标；第四类作为组合泛化指标；第五类不能仅靠接口随机化保证。

## 4. 非目标与约束

- 不立即删除当前 Stage 1--4、tool bank、toy pipeline 和 validators；它们作为 baseline
  和兼容回归路径保留。
- 不在第一阶段对现有 `outputs/` 做批量重写。
- 不让模型生成可直接执行的宿主 Python 代码；环境构造必须受 sandbox、allowlist 和资源限制。
- 不把 reference solution、private verifier、隐藏状态或完整 acceptance checklist 暴露给 rollout
  policy。
- 不以强制 15--30 个 tool call 作为 long-horizon 的主要准入条件。
- 不把 WikiHow 文本完全丢弃；它可以作为任务主题、用户目标和领域约束来源，但不再直接定义
  execution graph。
- `papers/` 保持只读；论文只作为设计依据。

## 5. 目标架构

```text
Verified Seed Task Bundle
        |
        v
Model: affordance scan + natural rewrite proposal
        |
        v
Transformation Contract
        |
        +--> Model: extend reference solution
        +--> Model: align environment/workspace
        +--> Model: derive outcome verifier
        +--> Model: write compact public instruction
        |
        v
Static / Contract / Anti-shortcut Gates
        |
        v
Fresh Sandbox: run reference solution + private verifier
        |
        v
Accepted Executable Task Bundle
        |
        +--> recursive reseeding
        +--> episode-specific tool rendering
        +--> counterfactual state variants
        |
        v
Hidden-environment Step-by-step Rollout
        |
        v
Execution Trace + State Deltas + Provenance DAG
        |
        v
Outcome / Causal / Branching / Minimality Validation
        |
        v
Full-trajectory SFT + Decision-point SFT + Preference/RL Tasks
```

## 6. 核心任务表示

### 6.1 Task Bundle

每个任务使用独立目录，建议结构如下：

```text
task_bundle/
  manifest.json
  instruction.md
  contract.json
  environment/
    environment.json
    fixtures/
  capabilities/
    bindings.json
  solution/
    reference_plan.json
    solve.sh                 # 仅 terminal 类型任务需要
  verifier/
    checks.json
    verify.py
  provenance/
    lineage.json
    generation.json
```

不同 domain 可以使用不同 runtime，但都必须映射到统一 bundle contract。

### 6.2 Public Instruction

公开 instruction 只包含：

- 用户目标；
- 不可从环境发现的硬约束；
- 公平的起始位置或探索入口；
- 主要 deliverable 或完成信号。

禁止包含：

- reference solution 的命令或 tool-call 顺序；
- private verifier 路径；
- 完整字段/文件清单；
- 内部 handle；
- 逐步 acceptance checklist。

### 6.3 Transformation Contract

`contract.json` 是生成阶段的内部真值，但不对 rollout policy 可见。建议字段：

```json
{
  "contract_version": "task-contract-v1",
  "goal": "user-visible semantic objective",
  "preserved_requirements": [],
  "new_requirements": [],
  "discoverable_evidence": [],
  "goal_predicates": [],
  "invariants": [],
  "intermediate_outcomes": [],
  "allowed_solution_families": [],
  "forbidden_shortcuts": [],
  "counterfactual_axes": [],
  "expected_reasoning_features": []
}
```

`expected_reasoning_features` 只能用于生成与事后审核，不能替代执行验证。

### 6.4 Goal Predicate

所有成功条件必须编译为可执行 predicate，例如：

```json
{
  "id": "release_is_safe_and_distributed",
  "all": [
    {"eq": ["$state.release.version", "3.4.0"]},
    {"eq": ["$state.release.channel", "beta"]},
    {"gte": ["$state.reports.tests.coverage", 85]},
    {"eq": ["$state.reports.security.critical", 0]},
    {"eq": ["$state.artifact.signed", true]},
    {"lte": ["$state.cost.total", 50]}
  ]
}
```

Verifier 必须报告每个 predicate 的通过状态和证据路径，以支持 partial credit 和诊断。

### 6.5 Invariant

Invariant 在全过程检查，而非只看最终状态：

- 未授权前不得执行不可逆写入；
- job 未成功前不得消费其 artifact；
- 成本不得超过预算；
- security failure 后不得发布旧 artifact；
- cancellation 后不得继续使用被取消 job 的输出。

### 6.6 Reference Solution

reference solution 的角色仅限于：

- 证明至少存在一条成功路径；
- 为 sandbox oracle validation 提供可执行 witness；
- 帮助定位 task/environment/verifier 不一致。

它不是 gold trajectory，也不直接进入训练集。Verifier 不得检查其具体动作顺序或临时实现细节。

## 7. Capability 与 Episode Tool 解耦

### 7.1 Internal Capability Registry

内部 capability 使用稳定语义和类型：

```json
{
  "capability_id": "entity.update.v1",
  "inputs": {
    "entity": "EntityHandle",
    "patch": "Object"
  },
  "outputs": {
    "entity": "EntityHandle",
    "revision": "RevisionHandle"
  },
  "preconditions": ["entity.exists", "permission.write"],
  "effects": ["entity.patch", "entity.revision.increment"],
  "errors": ["NOT_FOUND", "PERMISSION_DENIED", "CONFLICT"],
  "idempotency": "conditional",
  "risk": "reversible_write"
}
```

Capability registry 对 executor 可见，对 rollout policy 不可见。

### 7.2 Episode Tool Renderer

每个 episode 将 capability 渲染为具体 API。Renderer 支持：

- tool/argument/output 字段改名；
- flat 与 nested 参数互换；
- enum、boolean 和 tagged union 表达变化；
- error code 与 error object 表达变化；
- 一个 capability 拆成多个工具；
- 多个 capability 合并为一个受约束工具；
- 注入功能相近但约束不同的 distractor tools；
- 提供不同粒度的文档、example 或 `help` 工具。

每个 binding 必须保存从公开 API 到 internal capability 的可执行映射：

```json
{
  "public_name": "patch_asset",
  "capability_id": "entity.update.v1",
  "input_adapter": {},
  "output_adapter": {},
  "error_adapter": {}
}
```

### 7.3 Tool Discovery

为模拟 terminal 中的 `--help` 和本地文档，episode 可配置：

- 完整 JSON schema；
- 简短描述 + `describe_tool`；
- 示例调用；
- schema 部分缺省、执行后反馈类型错误；
- 多版本 API 文档。

不同 discovery 条件需要分层评测，避免把 schema 阅读困难与长程规划困难混为一谈。

## 8. Task Factory 流程

### 8.1 Seed 准入

首批 seed 必须是人工审核或现有 executable benchmark 中已验证的任务。每个 seed 至少包含：

- 可重建的初始环境；
- 公平的公开 instruction；
- private outcome verifier；
- 至少一条通过 verifier 的 reference solution；
- 无网络或资源不可控依赖，或已被本地 fixture 固定；
- 来源、许可和 lineage metadata。

当前 WikiHow 数据不能直接作为 executable seed，只能用于提出领域任务或扩展主题。

### 8.2 Affordance Scan

确定性扫描与模型排序共同识别 seed 可以自然支持的扩展，例如：

- 已存在配置和 manifest，可加入一致性修复；
- 已存在日志和失败状态，可加入诊断分支；
- 已存在异步 job，可加入 retry/cancel/timeout；
- 已存在权限、配额或锁，可加入约束驱动分支；
- 已存在多个工具或资源，可加入替代计划；
- 已存在中间 artifact，可加入 provenance 和最终汇总。

模型只能从扫描器允许的 operator/card 中选择，不能无约束地改变任务领域。

### 8.3 Rewrite Operator Taxonomy

初始 operator families 建议包括：

1. discovery and evidence;
2. configuration and consistency;
3. permission and authorization;
4. asynchronous lifecycle;
5. failure diagnosis and recovery;
6. resource and budget constraints;
7. artifact derivation and provenance;
8. rollback and idempotency;
9. multi-resource coordination;
10. alternative-plan affordance。

每个 operator card 明确定义：

- seed affordances；
- environment changes；
- expected state dependencies；
- verifier strategy；
- anti-shortcut strategy；
- counterfactual axes；
- 不适用条件。

### 8.4 Contract Generation

模型先生成 transformation contract，不直接改文件。确定性 gate 检查：

- 新 requirement 是否自然依赖 seed affordance；
- 所有 reward-relevant 条件是否公开或可发现；
- goal predicates 是否可编译；
- 是否存在非表面状态变化；
- 是否定义 shortcut rejection；
- 是否至少包含一个观察驱动决策候选；
- 是否可以构造至少一个合理 counterfactual variant。

### 8.5 Solution-First Rewrite

按受限顺序生成：

1. 扩展 reference solution；
2. 对齐 environment/fixtures；
3. 从 contract 独立生成 verifier；
4. 生成 compact instruction；
5. 运行跨组件 consistency pass。

“Solution-first”只表示先建立可执行 witness，不表示 verifier 应复制 solution。

### 8.6 Static Gates

在 sandbox 之前执行：

- bundle/schema 完整性；
- 文件写入 allowlist；
- dependency 和 fixture 可用性；
- goal/invariant predicate 可编译；
- verifier 不引用 reference solution 私有路径；
- instruction 不泄漏命令、handle 或完整 checklist；
- contract 的 discoverable evidence 确实存在；
- verifier 检查与 contract requirement 双向对齐；
- placeholder、stale artifact 和 hard-coded answer 检查；
- 与 parent 的差异不是仅修改 wording/verifier；
- build/time/memory/network policy 合规。

### 8.7 Sandbox Oracle Validation

在全新 sandbox 中：

1. 从初始状态构建环境；
2. 执行 reference solution；
3. 运行 private verifier；
4. 要求所有 goal predicates 通过且 invariant 未违反；
5. 保存执行日志、状态变化和 verifier evidence；
6. 对可修复错误最多进行有限轮 repair；
7. 每次 repair 后必须重建新 sandbox；
8. 持续失败的 candidate 丢弃，而不是弱化 verifier。

### 8.8 Contract Fairness Validation

任务只有同时满足以下条件才能 accepted：

- oracle validity：reference solution 在 fresh sandbox 中通过；
- contract validity：所有 reward-relevant requirement 均公开或可发现；
- anti-shortcut validity：已知 shortcut 不能通过；
- runtime validity：环境可稳定重建，没有偶发外部依赖。

## 9. Hidden-Environment Rollout

### 9.1 Policy 可见信息

rollout policy 只能看到：

- public instruction；
- observable workspace 或当前公开 state；
- episode-specific tools 和可用文档；
- 历史 action、tool response 和 user response；
- 公开的预算/时间等控制信息。

不得看到：

- `contract.json`；
- reference solution；
- private verifier；
- hidden state；
- internal capability ID；
- expected reasoning features；
- oracle plan 或 execution graph。

### 9.2 Episode Runner

runner 负责：

- 校验公开 tool call schema；
- 将 public tool 映射到 capability executor；
- 执行真实状态转移；
- 推进 logical clock 和 async event queue；
- 返回公开 observation；
- 记录隐藏 execution trace；
- 在每一步检查 invariant；
- 限制 token、step、cost、retry 和 wall-clock budget；
- 在 policy 请求结束时运行 private verifier。

### 9.3 User Simulator

user simulator 只提供工具无法发现的信息：

- 用户偏好；
- 风险确认；
- 私有 credential；
- 业务优先级；
- 冲突约束下的取舍。

它不能提供环境内部 handle、oracle 下一步或 verifier 条件。用户响应也应由独立规则或模型生成，
并经过 consistency check。

### 9.4 Async Runtime

异步对象至少支持：

```text
queued -> running -> succeeded
                  -> failed(retryable)
                  -> failed(terminal)
       -> cancelled
```

状态推进可以由 logical time、poll 次数或其他事件触发。Executor 必须拒绝：

- 在 artifact 未产生时消费 artifact；
- 对 terminal job 重试而没有新输入；
- 使用 cancelled job 的输出；
- 超过 retry policy；
- 在成本不足时继续启动 job。

### 9.5 Rollout 终止

policy 的自然语言“已完成”不代表成功。Episode 状态分为：

- `goal_satisfied`；
- `terminal_failure`；
- `invariant_violation`；
- `budget_exhausted`；
- `agent_stopped_incomplete`；
- `runtime_error`。

训练数据必须保留终止原因，不能把基础设施错误混入策略失败。

## 10. Execution Trace 与 Provenance

每次 action 生成结构化 trace：

```json
{
  "step": 12,
  "public_tool": "patch_asset",
  "capability_id": "entity.update.v1",
  "arguments": {
    "asset_ref": {
      "value": "asset_91",
      "source": {"step": 5, "output_path": "$.artifact.handle"}
    }
  },
  "selected_branch": "permission_granted",
  "read_set": ["$state.assets.asset_91", "$state.permissions.write"],
  "write_set": ["$state.assets.asset_91.version"],
  "produced_handles": ["revision_7"],
  "consumed_handles": ["asset_91"],
  "goal_delta": [],
  "invariant_results": []
}
```

Provenance 由 executor 产生，不能依赖 LLM 自报。它用于构建：

- argument provenance graph；
- state read/write dependency graph；
- handle production/consumption graph；
- observation-to-decision graph；
- goal evidence graph。

## 11. 因果与开放性验证

### 11.1 Outcome Validation

- 最终 goal predicates 全部求值；
- 每个 predicate 保存实际值和 evidence path；
- invariant 在整个 trace 上求值；
- verifier 支持 binary success 和 dense partial credit；
- verifier 不依赖固定 tool 名或调用顺序。

### 11.2 Delayed Handle Validation

验证早期产生的 handle 是否在至少 `k` 个有意义动作之后被消费。中间仅包含自然语言或重复
poll 不计入距离。建议 long tier 至少存在一个 `distance >= 4` 的 handle dependency。

### 11.3 Derived Object Chain

要求存在至少一条多阶段对象链：

```text
source handle -> job handle -> artifact handle -> report handle -> final object
```

每条边必须来自 executor provenance，而不是字符串相同。

### 11.4 Observation-Dependent Decision

对关键 observation 执行干预：

- 权限 granted -> denied；
- quota sufficient -> insufficient；
- job success -> retryable failure；
- target exists -> missing；
- candidate A available -> unavailable。

使用相同历史前缀重新询问 policy 或 oracle planner。只有当正确后续动作集合随 observation 改变，
且 rollout 采取相应分支时，才计为 observation-dependent decision。

### 11.5 Action Ablation 与 Necessary Action Ratio

依次删除或替换候选 action，并从相应 state snapshot 重放后续步骤。如果目标仍能无修改通过，
该 action 不属于必要因果链。

```text
necessary_action_ratio = necessary_actions / non-communication_actions
```

long tier 不要求每个动作都必要，但低比例轨迹应降权或拒绝。

### 11.6 Semantic Recovery

失败恢复必须满足：

- error 有稳定 class；
- recovery action 修改 error 所涉及的 precondition、输入或资源；
- 随后的成功属于同一 subgoal；
- 不能用无关成功响应冒充 recovery；
- terminal error 后继续盲目重试应被判错。

### 11.7 Goal-Grounded Final Verification

最后的验证必须读取或计算 goal predicate 使用的 state paths。仅调用 `verify_resource` 或返回
`status=success` 不够。Verifier 需要检查最终 observation 的 evidence 是否覆盖原始目标。

### 11.8 Alternative Solution Validation

使用不同 planner、不同模型温度或显式约束搜索成功轨迹。将轨迹投影到 capability/state-effect
层，而不是比较公开 tool 名。至少保留：

- 两条 effect-equivalent 但动作组织不同的成功计划；或
- 一条主计划及一个经状态干预后成立的替代计划。

若所有成功解都与 reference solution 高度同构，任务可用于基本 SFT，但不进入 high-branching tier。

### 11.9 Effective Branching Factor

在关键状态采样候选 action，并在 sandbox snapshot 上执行，按结果分类：

- `progressing`：提高 verifier progress；
- `information_gaining`：暴露后续决策需要的信息；
- `recoverable`：不推进目标但状态仍可恢复；
- `neutral`：无有效影响；
- `irreversible_failure`：破坏 invariant 或使目标不可达。

建议记录而不追求单一越大越好的分数：

```text
valid_action_count
distinct_progress_effects
decision_entropy
recoverable_branch_count
irreversible_trap_count
```

高 branching 任务要求存在多个语义不同的合理动作，而不是仅暴露大量同义工具。

## 12. 反捷径与泄漏检查

至少实现以下 adversarial baselines：

- 不检查环境、直接写最终答案；
- 复制 stale artifact；
- 创建空文件或 placeholder；
- hard-code verifier 期望值；
- 跳过中间 derivation；
- 仅执行 reference solution 的最后一步；
- 读取 private verifier/reference solution；
- 使用隐藏 handle；
- 用工具名猜固定调用模板；
- 重复 no-op 直到达到长度阈值。

任务只有在这些 shortcut policies 失败时才通过 anti-shortcut gate。

## 13. 数据产品

同一 accepted task 生成多种训练数据，而不是只输出完整 SFT trajectory。

### 13.1 Full-Trajectory SFT

保留成功 hidden rollout。训练时：

- assistant natural text 和 tool call 可计算 loss；
- tool response 作为环境 context，不作为模型应生成的目标；
- reference solution 不进入记录；
- metadata 保存 task/variant/tool-renderer IDs，但训练模板可选择隐藏。

### 13.2 Decision-Point SFT

从 observation-dependent decision 前截取前缀，目标仅为下一步 action。这样能够提高关键因果决策
的训练权重，避免长轨迹中机械步骤淹没稀少的分支决策。

### 13.3 Preference Pairs

构造 chosen/rejected：

- 正确消费 handle vs 猜测隐藏 handle；
- 根据权限切换计划 vs 继续原写入；
- retryable failure 后修复再重试 vs 原参数盲重试；
- goal-grounded verification vs 空泛完成声明；
- 有效替代工具 vs 与约束冲突的工具。

### 13.4 Verifier-Based RL Tasks

accepted bundle 本身作为 RL task。Reward 建议由：

- goal predicate progress；
- invariant preservation；
- shortcut penalty；
- cost/step budget；
- terminal outcome

共同构成，避免只奖励最终二值成功。

### 13.5 Failure Corpus

保留可解释失败，而不是全部丢弃：

- incorrect plan；
- hidden-state assumption；
- wrong tool interpretation；
- async misuse；
- recovery failure；
- premature termination；
- infrastructure failure。

基础设施失败必须单独隔离，不能作为 rejected policy example。

## 14. 目录与模块规划

建议新增以下目录，不直接塞入现有单文件脚本：

```text
gem_repro/
  task_factory/
    models.py
    manifests.py
    seed_loader.py
    affordance_scan.py
    operator_registry.py
    contract_builder.py
    bundle_rewriter.py
    consistency.py
    sandbox.py
    reseed.py
  runtime/
    capability_registry.py
    tool_renderer.py
    adapters.py
    executor.py
    predicates.py
    async_jobs.py
    snapshots.py
    provenance.py
  rollout/
    policy_client.py
    episode_runner.py
    user_simulator.py
    termination.py
    trace_recorder.py
  causal_validation/
    outcome.py
    contract.py
    leakage.py
    shortcuts.py
    dependencies.py
    interventions.py
    alternatives.py
    branching.py
    minimality.py
  schemas/
    task_manifest_v1.json
    task_contract_v1.json
    capability_v1.json
    tool_binding_v1.json
    episode_v1.json
    execution_trace_v1.json
    validation_report_v1.json
  prompts/
    task_affordance_rank.txt
    task_contract_generate.txt
    reference_solution_rewrite.txt
    outcome_verifier_generate.txt
    public_instruction_align.txt
    task_bundle_repair.txt
    counterfactual_variant_generate.txt
    hidden_rollout_policy.txt
  scripts/
    build_task_seeds.py
    synthesize_task_bundles.py
    validate_task_bundles.py
    render_task_episodes.py
    rollout_task_episodes.py
    validate_causal_episodes.py
    export_task_training_data.py
    analyze_task_factory.py
  tests/
    fixtures/
    test_contracts.py
    test_predicates.py
    test_tool_renderer.py
    test_executor.py
    test_async_jobs.py
    test_provenance.py
    test_interventions.py
    test_branching.py
```

## 15. 现有代码的处置

### 15.1 保留为 Baseline

第一阶段不修改或删除：

- `scripts/toy_gem_pipeline.py`；
- `scripts/build_llm_requests.py` 的现有 Stage 1--4；
- `scripts/executable_environment.py`；
- `scripts/validate_trajectories.py`；
- `config/tool_bank.json`；
- `legacy_no_validation/`。

### 15.2 后续适配

- `llm_client.py` 和 `execute_llm_requests.py` 可抽取为共享 provider 层；
- `rollout_stage3.py` 的 step-by-step 交互逻辑迁移到 `rollout/episode_runner.py`；
- `quality_gate.py` 扩展或新增 task-first quality gate，不改变旧参数语义；
- `convert_to_sft.py` 保留旧格式，新 exporter 输出 task-first 数据；
- `tool_bank.json` 仅作为一个固定 renderer preset，不再作为所有任务的语义真值；
- `text-exec-dsl-v0` 用于 baseline fixture，新 runtime 使用独立版本和 adapter。

## 16. 实施阶段

### Phase 0：冻结 Baseline 与建立实验协议

交付物：

- 固定一批当前 pipeline 的代表性输入与最终 artifact；
- 记录现有 yield、tool-call、execution pass 和数据分布；
- 选择 20--50 个 executable seed task；
- 定义 task-first 实验 split 和随机种子；
- 建立旧 pipeline smoke regression。

退出标准：旧 baseline 可一条命令复现；新工作不影响旧输出。

### Phase 1：Schema、Predicate 与 Bundle Loader

实现：

- 所有 v1 JSON schema；
- task bundle loader/manifest；
- goal predicate 和 invariant evaluator；
- verifier evidence report；
- 3--5 个手写 executable fixtures。

退出标准：fixtures 的成功、失败和 partial credit 可确定性复现。

### Phase 2：Capability Runtime 与 Tool Renderer

实现：

- typed handles；
- capability registry；
- input/output/error adapters；
- deterministic state transition；
- renderer rename/nesting/error variants；
- distractor tools；
- split/merge 的有限模板。

退出标准：同一 task 在至少三种 API render 下得到等价 state outcome，公开 schema 不泄漏
internal capability ID。

### Phase 3：Hidden Rollout 与 Trace

实现：

- episode runner；
- policy context boundary；
- snapshot/reset；
- async lifecycle；
- user simulator boundary；
- structured provenance trace；
- terminal status 分类。

退出标准：policy 无法访问 solution/verifier/hidden state；每个关键参数能追溯来源；异步和 invariant
错误会被 executor 拒绝。

### Phase 4：Task Factory MVP

实现：

- seed affordance scan；
- 10 个左右 rewrite operators；
- transformation contract generation；
- solution/environment/verifier/instruction 分阶段生成；
- static gates；
- sandbox oracle validation；
- bounded repair。

退出标准：从至少 20 个 seed 生成一批 oracle-valid、contract-valid 的 children，且 shortcut
baseline 不能通过。

### Phase 5：Causal 与 Branching Validation

实现：

- delayed handle；
- derived object chain；
- semantic recovery；
- goal-grounded verification；
- action ablation；
- observation intervention；
- alternative solutions；
- effective branching metrics。

退出标准：手工构造的单轨、伪恢复、冗余长链和 verifier-only task 均被准确拒绝或降级。

### Phase 6：Counterfactual Variants 与 Recursive Reseeding

实现：

- state variant generator；
- variant solvability validation；
- parent/lineage manifest；
- domain/operator/parent caps；
- difficulty-aware reseeding；
- near-duplicate 和 lineage collapse 检查。

退出标准：同一 contract 的多个 variant 在关键 observation 后产生不同正确策略；连续 2--3 轮
递归仍保持 validation yield 和 operator diversity。

### Phase 7：Training Export 与消融实验

实现：

- full trajectory exporter；
- decision-point exporter；
- preference pair exporter；
- RL task packaging；
- model-specific tool template adapters。

实验至少比较：

1. 当前 tool-bank one-shot；
2. 固定 API + hidden rollout；
3. randomized API + hidden rollout；
4. task-first + outcome validation；
5. task-first + causal/branching gates；
6. 加入 counterfactual pairs。

退出标准：在 held-out API 和 held-out capability composition 上取得稳定提升，而不是只提升同模板
validation pass rate。

## 17. 测试策略

### 17.1 单元测试

- predicate 的所有 operator 和缺失路径；
- invariant 的逐步检查；
- handle type safety；
- adapters 的双向转换；
- async state machine；
- snapshot 隔离；
- provenance read/write set；
- verifier evidence coverage。

### 17.2 集成测试

- reference solution 在 fresh sandbox 通过；
- shortcut solution 失败；
- public instruction 删除某硬约束后 contract gate 失败；
- tool renderer 改名后任务仍可完成；
- observation 干预后原计划失效、替代计划成功；
- repair 后必须重新 sandbox 验证；
- recursive child 不修改 allowlist 外文件。

### 17.3 回归测试

- 当前 toy pipeline；
- 当前 validators；
- task-first fixtures；
- schema backward-compatibility；
- provider-free deterministic tests 默认可运行。

## 18. 质量指标与准入分层

### 18.1 基础指标

- synthesis candidate pass rate；
- oracle-valid rate；
- contract-valid rate；
- anti-shortcut pass rate；
- infrastructure failure rate；
- rollout success/partial-credit rate；
- API renderer coverage；
- domain/operator/lineage diversity。

### 18.2 因果指标

- critical dependency depth；
- delayed handle distance；
- derived object chain length；
- observation-dependent decision count；
- semantic recovery count；
- async decision count；
- necessary action ratio；
- goal evidence coverage；
- counterfactual branch accuracy。

### 18.3 Branching 指标

- distinct progressing effects per key state；
- alternative successful plan count；
- decision entropy；
- valid/recoverable/terminal action 分布；
- reference-solution similarity of successful rollouts。

### 18.4 数据分层

- **Executable**：oracle、contract、runtime 通过；
- **Grounded**：无泄漏、provenance 完整、goal evidence 完整；
- **Causal**：达到依赖深度、必要动作和语义恢复标准；
- **Branching**：存在替代策略或通过 observation intervention；
- **Long-Horizon Gold**：同时满足 Causal、Branching、async/constraint 特征及质量预算。

不要把所有 executable task 都称为 long-horizon gold。

## 19. 推荐的 MVP 验收阈值

首个 100-task MVP 可采用保守阈值：

- oracle-valid task >= 60；
- contract-valid / oracle-valid >= 90%；
- shortcut rejection >= 95%；
- infrastructure error <= 5%；
- 每个任务至少 2 种 API render；
- causal tier 至少 30 个任务；
- causal tier 的 dependency depth >= 6；
- 至少一个 delayed handle distance >= 4；
- 至少一个 observation-dependent decision；
- goal evidence coverage = 100%；
- necessary action ratio >= 0.6；
- branching tier 至少 15 个任务；
- held-out tool-name/schema 评测相对固定 tool-bank baseline 有显著提升。

这些阈值是 MVP gate，后续应根据实际分布调整，不能通过 prompt 强行制造数字。

## 20. 实验 Split 设计

避免随机 row split 造成模板泄漏，至少建立：

- unseen public tool names；
- unseen argument/output schemas；
- unseen renderer families；
- unseen rewrite operators；
- unseen capability compositions；
- unseen parent lineages；
- unseen domains；
- counterfactual state variants held out。

评测同时报告 task success、partial credit、step/cost、invariant violation 和 recovery quality。

## 21. 风险与缓解

### 21.1 Verifier 被 Reference Solution 污染

缓解：verifier 从 contract 独立生成；加入 cross-model review；比较 verifier checks 与 solution
incidental artifacts；运行替代成功解。

### 21.2 递归只增加机械工作

缓解：operator 选择要求新增 state dependency 或 observation decision；使用 action ablation、
dependency depth 和 solver 行为，而不是 solution 行数决定 difficulty。

### 21.3 环境过于人工或狭窄

缓解：从真实 executable seeds 出发；限制环境修改规模；要求 alternative plan 或 state
intervention；审核 affordance 数量和 shortcut。

### 21.4 Tool Renderer 产生无意义随机化

缓解：所有 renderer 必须保持 capability contract；分别测试 rename、schema、split/merge；禁止只靠
随机字符串制造表面差异。

### 21.5 模型与 Verifier 合谋式自洽

缓解：生成、review、rollout 使用不同 prompt 或模型；确定性 static gates；adversarial shortcut
policies；sandbox 执行；held-out verifier audit。

### 21.6 Sandbox 成本过高

缓解：先做 static preflight；使用 copy-on-write snapshot；缓存环境 build；限制 repair；先在小 seed
集验证 yield，再扩展规模。

### 21.7 小模型被接口理解负担压垮

缓解：使用 curriculum，从完整 schema 到部分 discovery；混合固定和变体接口；单独训练 tool
documentation comprehension 与 long-horizon planning。

## 22. 关键设计决策

实施前需要固定以下默认选择：

1. **保留旧 pipeline**：是，直到 task-first 在 held-out benchmark 上优于 baseline。
2. **默认生成单位**：task bundle，不是 trajectory。
3. **reference solution 是否训练**：否，仅作 oracle proof。
4. **公开 execution graph**：否。
5. **默认 rollout**：hidden environment、step-by-step。
6. **成功判据**：private outcome verifier + invariant checks。
7. **tool bank 角色**：一个 renderer preset，不是统一语义空间。
8. **Stage 4 角色**：逐步退化为 counterfactual/task evolution，不再主要负责消息扩写。
9. **长程定义**：因果依赖深度和观察驱动决策，不是 call count。
10. **递归 reseeding**：只使用 oracle-valid、contract-valid、anti-shortcut-valid tasks。

## 23. 第一批具体开发任务

建议按以下顺序开始编码：

1. 新增 v1 schemas 和 3 个手写 task fixtures；
2. 实现 predicate/invariant evaluator 与 evidence report；
3. 实现 capability registry、typed handle 和两个 tool renderer；
4. 实现 sandbox snapshot、hidden episode runner 和 structured trace；
5. 实现 outcome、leakage、provenance validators；
6. 实现 delayed handle、semantic recovery、goal-grounded verification；
7. 实现一个 observation intervention 和 action ablation prototype；
8. 用手写 bundles 收集第一批 rollout，验证数据格式；
9. 再实现模型驱动的 contract/solution/verifier generation；
10. 最后加入 recursive reseeding，避免在基础 runtime 未稳定前扩大生成规模。

## 24. 最终验收原则

重构成功不能只由“生成了更长轨迹”或“sandbox verifier 通过率高”来判断。最终至少需要证明：

- agent 在看不到 oracle 的情况下能够通过探索完成任务；
- verifier 检查语义结果而非固定路径；
- observation 改变时，正确策略也随之改变；
- 关键参数和 handle 具有机器可读 provenance；
- 删除关键动作会破坏目标或后续可执行性；
- 至少一部分任务存在不同的成功计划；
- 改变工具名称和 schema 后，训练收益仍能迁移；
- 小模型在 held-out capability composition 上优于当前固定 tool-bank 数据训练结果。

只有满足这些条件，数据才真正从“可回放的人工 workflow”升级为“模型生成、环境开放、因果可验证的长程交互任务”。
