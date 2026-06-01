# Atomic Tool Bank 修改流程记录

本文档记录当前 GEM 数据生成 pipeline 的就地修改计划。我们不另开版本号、不维护一套并行 pipeline，而是在现有 Stage2 / Stage3 / validation 基础上直接改。这个文件用于记录修改动机、设计约束、实施顺序和后续需要持续讨论的问题。

## 修改动机

当前 pipeline 已经可以生成 replay-valid 的 tool-use 数据，但数据形态还有两个明显问题：

- 轨迹偏短，user 交互偏多，assistant 自主连续调用工具的比例不够。
- 工具粒度偏顶层、偏任务专用，例如不同任务里会出现 `login_student`、`authenticate_scanner`、`upload_file_to_cloud` 这类一次性完成大动作的工具。

这会导致模型学到“看到任务就调用某个任务专用 API”，而不是学习在通用工具空间里做长程规划、状态读取、错误恢复和结果验证。

接下来的修改目标是：

- 让 tool-call chain 更长，user turn 更少。
- 让工具在不同任务间复用，而不是每条数据临时自造一套工具。
- 增加状态检查、规划、验证、恢复等中间步骤。
- 减少任务专用的高层工具。
- 把看起来是单步、但实际复杂的动作拆成多个原子工具调用。

## 核心决策

Stage2 不应该继续自由发明每条样本自己的工具命名空间。Stage2 应该从一个 canonical tool bank 里选择工具，并围绕这些工具生成该样本专属的 workflow、initial state 和 executable tool rules。

tool bank 不是封闭世界假设。它是一个受控注册表：新工具可以加入，但必须通过明确的 admission 流程，而不是在单条样本里随意发明。

## Tool Bank 如何初始化

初始 tool bank 不需要一次性完美覆盖所有真实应用。它应该从小而可控的 reusable primitive 集合开始，然后通过数据生成过程中发现的缺口逐步扩展。

初始化来源有三类：

1. computer-use 任务中高频出现的原子操作。
2. 当前已有 artifact 中生成过的高层工具的抽象拆解。
3. 跨任务、跨领域都能复用的数据 / 对象操作。

初期建议优先使用“语义原子工具”，而不是直接降到 pixel / click 级别。这样 replay environment 仍然可控，同时也能显著拉长轨迹并增加推理步骤。

建议的初始工具类别如下。

### Navigation and Discovery

- `open_resource`
- `search_records`
- `list_records`
- `get_record`
- `read_state`

### Authentication and Permissions

- `authenticate`
- `set_permission`
- `verify_permission`

### File and Resource Handling

- `locate_file`
- `create_upload_session`
- `upload_resource`
- `get_share_link`
- `export_resource`
- `download_resource`
- `verify_resource`

### Record Mutation

- `create_record`
- `update_record`
- `attach_resource`
- `delete_record`

### Communication

- `send_message`
- `add_recipient`
- `verify_delivery`

### Long-running Operations

- `start_job`
- `poll_job`
- `cancel_job`
- `verify_job_result`

初始 tool bank 应该足够小，方便人工检查。目标不是立刻覆盖所有应用，而是先逼迫 Stage2 用可复用 primitive 表达任务流程。

## 如果实际应用必须自造工具怎么办

这是合理情况。真实任务中确实会出现现有 tool bank 覆盖不了的能力。关键是不能让 Stage2 在最终 artifact 里直接自由使用新工具，而应该走一个受控的缺口发现和工具准入流程。

建议行为：

1. Stage2 首先尝试只用现有 canonical tools 解决任务。
2. 如果无法表达任务，Stage2 输出 `missing_tool_requirements`，描述缺失能力。
3. 该样本暂时不进入 Stage3 轨迹生成，也不转成 SFT。
4. 人工或自动 review 判断这个缺失能力属于哪一类：
   - 其实可以由已有工具组合表达；
   - 是值得加入 tool bank 的可复用原子能力；
   - 太领域专用，应该拒绝；
   - 应该被表达成 environment state / domain data，而不是新增工具。
5. 如果通过准入，就把该工具加入 canonical tool bank，然后重新生成受影响的 Stage2 样本。

这样做的结果是：tool bank 可以增长，但增长是可解释、可追踪、可复用的，不会退化成每条样本一套任务专用高层 API。

## 新工具准入标准

一个新工具通常需要满足多数条件，才应该进入 tool bank：

- 能在多个任务或多个领域中复用。
- 执行一个原子操作，或者一个稳定的平台 primitive。
- 名称不绑定某个 WikiHow 标题或某个具体任务。
- 参数足够通用，可以覆盖多个对象或服务。
- 输出可以被 `text-exec-dsl-v0` 确定性 replay。
- 无法被已有工具的短组合自然表达。

以下类型的工具通常应该被拒绝或拆解：

- `login_student`
- `scan_to_email`
- `create_assignment_with_link`
- `download_offline_map`
- `export_gpx_track`

这些更适合作为 workflow，由多个通用工具组合完成。

## 例子：如何拆解高层动作

### Login

避免任务专用工具：

- `login_student`
- `authenticate_scanner`
- `dreambox_login`

优先改成通用工具：

- `authenticate(service, account_id, credential_type, credential_value)`

如果需要更细，可以拆成：

- `open_resource(service_or_url)`
- `get_record(collection, query)`
- `authenticate(...)`
- `verify_session(session_id)`

### Upload and Attach File

避免：

- `upload_file_to_cloud`
- `create_assignment_with_link`

优先拆成：

- `locate_file(file_name)`
- `create_upload_session(provider)`
- `upload_resource(session_id, file_id)`
- `set_permission(resource_id, visibility)`
- `get_share_link(resource_id)`
- `create_record(collection, fields)`
- `attach_resource(target_type, target_id, resource_id)`
- `verify_record(collection, record_id)`

### Offline Map Download

避免：

- `download_offline_map(location_id)`

优先拆成：

- `list_records(collection="map_catalog", query=...)`
- `get_record(collection="map_region", id=...)`
- `check_constraint(type="storage", required=...)`
- `start_job(job_type="download", target_id=...)`
- `poll_job(job_id)`
- `update_record(collection="settings", fields=...)`
- `verify_resource(resource_id)`

## Stage2 修改方向

Stage2 prompt 需要在当前文件基础上直接改，使它：

- 接收 canonical tool bank 作为上下文。
- 只能从 tool bank 中选择工具。
- 用原子工具组织 workflow steps。
- 为选中的工具生成 `environment.initial_state` 和 `environment.tool_rules`。
- 当 tool bank 不足时输出 `missing_tool_requirements`，而不是直接自造工具。
- 避免任务专用的高层 tool name。

Stage2 artifact 仍然保留当前字段：

- `workflow`
- `tools`
- `environment`

但 `tools` 应该来自 tool bank，而不是每条样本自由发明。

## Stage3 修改方向

Stage3 prompt 也在当前文件基础上直接改，使它：

- 减少初始请求之后的 user turn。
- 鼓励 assistant 通过工具读取状态，而不是频繁问用户要信息。
- 在 environment 支持时生成更长 tool-call chain。
- 成功前执行 postcondition verification。
- 当 environment 暴露真实不可恢复约束时，允许最终失败。
- 避免为了变长而重复无意义 no-op 工具调用。

建议的轨迹长度分布：

- short：3-6 个 tool calls。
- medium：7-14 个 tool calls。
- long：15-30 个 tool calls。

这个分布应该在 batch 层面控制，不应该机械要求每条样本都很长。

## Validator 修改方向

需要新增或扩展 validator：

- `canonical_tool_only`：所有工具必须来自 tool bank，除非样本明确进入 missing-tool proposal 状态。
- `composite_tool_lint`：拒绝任务专用的高层工具名。
- `min_tool_calls`：按轨迹类型限制最低 tool-call 数。
- `max_user_turns`：限制 user 交互次数。
- `state_dependency_check`：后续参数尽量依赖前面 tool 输出，而不是直接凭空填。
- `postcondition_check`：成功轨迹在环境支持时必须包含最后的 read/get/verify 工具。
- `missing_tool_handling`：带 `missing_tool_requirements` 的样本不能转成 SFT。

已有 validator 继续保留：

- `validate_environment.py`：检查 executable sandbox。
- `validate_tool_bank.py`：检查 canonical tool bank、function schema 和 discoverability。
- `validate_trajectories.py --strict-grounding`：检查 grounding 和消息结构。
- `validate_execution.py`：检查 DSL replay。

## 实施顺序

1. 在当前 repo 中新增 canonical tool bank 文件。
2. 修改 Stage2 request building，把 tool bank 注入 Stage2 prompt。
3. 直接改写现有 Stage2 prompt，要求只能选择 tool bank 中的工具。
4. 新增 validator，拒绝非 tool-bank 工具，除非它是 missing-tool proposal。
5. 直接改写现有 Stage3 prompt，让轨迹更长、user turn 更少、包含验证步骤。
6. 新增轨迹长度和 user-turn validator。
7. 用现有 10 条样本重新跑小规模实验，对比：
   - 平均 tool-call 数；
   - user turn 数；
   - replay pass rate；
   - tool reuse rate；
   - missing-tool proposal 数。
8. 小规模稳定后再恢复更大规模生成。

## 当前开放问题

- 初始 tool bank 应该控制在多小？
- 是否只使用语义原子工具，还是也加入 `click`、`type_text`、`read_screen` 这类 UI-like 工具？
- `missing_tool_requirements` 初期是否完全人工 review，还是写一个自动聚类 / 准入脚本？
- 最终失败样本是否应该默认跳过 workflow-coverage 检查，还是显式加 trajectory outcome 标签？

## 当前立场

tool bank 应该从小型、可复用、语义原子的 primitive 集合开始，通过受控 admission 逐步扩展。如果真实应用确实需要 custom tool，pipeline 应该支持新增，但前提是它足够原子、可复用、可 replay。这样既保留扩展性，又能避免数据重新退化成一次性高层任务 API。
