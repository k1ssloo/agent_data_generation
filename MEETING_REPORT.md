输入数据来自 WikiHow computer-use 文本，每条样本包含标题与步骤。大规模运行由 `scripts/run_pipeline_shards.py` 负责：它把输入 JSONL 切成固定大小 shard，逐个调用 `scripts/run_pipeline.py`，每个 shard 成功后聚合 `sft_openai_messages.jsonl` 与 `shard_summary.json`。这样做的作用是支持断点续跑、低产 shard 监控和失败隔离。

### Stage1: Text Filtering

Stage1 使用 LLM 判断一条文本是否适合转成多步 tool-use agent 任务，并抽取 `summary/domain/platform/task`。

关键 prompt 节选来自 `prompts/stage1_filter.txt`：

```text
Determine whether the following text contains multi-step operations involving
the use of an app, website, computer, or other machine.
```

### Stage2: Tool Bank + Executable Environment

Stage2 是当前 pipeline 的关键。它不允许模型随意发明任务专属大工具，而是要求从 canonical tool bank 中选择原子工具，并生成：

- `workflow`: 任务描述、步骤、execution graph
- `tools`: OpenAI function schema，只能复制 canonical tool bank 中的工具
- `environment`: `text-exec-dsl-v0` 环境，包括 `initial_state` 和 `tool_rules`

关键 prompt 节选来自 `prompts/stage2_workflow_tools.txt`：

```text
Convert each important action into a sequence of canonical atomic tools from the tool bank.
Do not invent task-specific tools when the same capability can be represented by existing canonical tools.
```

```text
environment.version must be exactly "text-exec-dsl-v0".
environment.initial_state is a JSON object containing concrete synthetic entities.
environment.tool_rules must contain exactly one rule for every tool name.
```

这里的 `tool_rules` 是可执行性的来源。每个工具都有分支条件、返回值和可选 state effects，例如读取记录、创建记录、更新状态、上传资源、启动异步任务等。Stage2 之后会运行 `validate_environment.py` 和 `validate_tool_bank.py --require-discoverable-record-ids`，检查 DSL 语法、tool rule 覆盖、工具 schema 以及隐藏 ID 是否能通过 list/read/search 等前置工具被发现。

### Stage3: Trajectory Generation

Stage3 使用 source text、workflow、tools 和 executable environment 生成多轮对话。它要求 assistant 必须按 OpenAI message 格式调用工具，并且参数必须由用户、assistant 可见文本或前序 tool response ground。

关键 prompt 节选来自 `prompts/stage3_trajectory.txt`：

```text
Tool responses must be derivable from the executable environment state and tool_rules DSL.
Do not invent an entity, state change, ID, option, or failure that is not supported by the environment.
```

```text
Prefer long-horizon tool use when the workflow and environment support it.
Target 7 to 14 tool calls when natural.
```

Stage3 仍是 one-shot 轨迹生成，而不是让模型在真实环境中一步步 rollout。当前更实际的路线是：允许模型先生成完整 response，然后用 replay engine 统一重放和纠错。

### Tool Response Canonicalization 与 Replay Engine

`scripts/canonicalize_tool_responses.py` 会遍历每条 assistant tool_call，调用 `scripts/executable_environment.py` 中的 `execute_tool()`，用 DSL environment 的当前 state 和 tool_rules 计算 expected response，然后替换原始 tool message。

执行逻辑核心是：

1. 从 row 中读取 `environment.initial_state`
2. 每遇到一个 assistant tool_call，找到同名 `tool_rules`
3. 按 branch 顺序判断 `if` 条件
4. 生成 `response`
5. 应用 `effects` 修改 state
6. 把下一条 tool message 的 content 改成 replay 得到的 response

这一步解决了早期方案最大的问题：早期数据中的 tool response 是模型自然语言伪造的，不一定和任何可执行状态一致；当前数据中的 tool response 至少要能由 DSL state 转移推出。

### Stage3 Repair

Stage3 之后会同时跑三类验证：

- `validate_trajectories.py`: strict grounding、消息格式、工具 schema、workflow tool coverage、错误恢复、final verification
- `validate_execution.py`: 用 replay engine 再执行一遍，检查 response 与 state effects
- `validate_tool_bank.py`: 检查工具是否仍符合 canonical atomic tool bank 与可发现 ID 约束

失败样本会进入 `stage3_repair`，最多两轮。关键 prompt 来自 `prompts/stage3_repair_trajectory.txt`：

```text
Repair a generated multi-turn tool-use trajectory so it passes strict grounding
and executable environment replay.
```

```text
If execution errors are caused by the environment/tool_rules instead of the message sequence,
return minimally repaired workflow/tools/environment along with messages.
```

这意味着 repair 不只是改对话，也可以在必要时最小修改环境规则，使轨迹和 environment 一致。

### Stage4: Refinement

在 Stage3 已通过样本上增加复杂度，而不是重新生成全新任务。它只能修改 messages 和 refinement metadata，不能改工具或环境。

关键 prompt 节选来自 `prompts/stage4_refine.txt`：

```text
Rewrite the original trajectory to make it more complex, realistic, and useful
for agent training while preserving executable validity.
```

```text
Increase complexity through non-trivial tool-call chains, realistic user ambiguity,
prerequisite checks, dependency on prior tool outputs, error recovery, conditional logic,
and final verification.
```

Stage4 的目标包括 `long_horizon_dependency`、`state_inspection`、`clarification`、`error_recovery`、`conditional_branch`、`constraint_refusal` 和 `final_verification`。Stage4 之后仍然会 replay、strict validate、execution validate、tool-bank validate，并在失败时进入同一套 trajectory repair。若 Stage4 复杂化后仍失败，`run_pipeline.py` 会用 `fallback_invalid_rows()` 回退到 Stage3 已验证版本，避免为了复杂度牺牲有效率。

###SFT Export

最终通过三类验证的样本会进入 `convert_to_sft.py`，输出为 OpenAI messages 格式。SFT 文件保留 `id/source_text/tools/messages/metadata`，其中 `metadata` 包含 summary、domain、task、workflow 和 refinement patterns 等信息。

## 当前数据质量

基于 `outputs/runs/wikihow_computer_10k/sft_openai_messages.jsonl` 的统计：

| 指标 | 当前结果 |
| --- | --- |
| records | 3261 |
| messages | min 9 / median 19 / p90 27 / max 55 |
| tool calls | min 3 / median 7 / p90 10 / max 26 |
| unique tools | min 2 / median 5 / p90 7 / max 11 |
| user turns | min 1 / median 1 / p90 2 / max 6 |
| 含读取/验证类工具 | 3227 条 |

高频工具已经从早期任务专属工具转向 canonical atomic tools：`open_resource`、`list_records`、`create_record`、`update_record`、`verify_resource`、`get_record`、`start_job`、`search_records`、`poll_job`、`read_state`、`locate_file`、`authenticate` 等。

总体质量判断：

- 优点：工具更原子，response 可由 state replay，常见 CRUD、上传、下载、异步 job、final verification 都已覆盖。
- 优点：Stage4 和 repair 让部分轨迹具备更自然的状态检查、失败恢复和多轮确认。
- 风险：DSL environment 是模型生成的轻量环境，不等价于真实网站或真实 app；它解决的是内部一致性，不解决真实世界 API 完整性。
- 风险：Stage1 通过率在部分 shard 上波动，可能需要分 domain 诊断。
- 风险：SFT 中不保留完整 environment，后续若要做执行复验，需要回到 stage artifacts 或 shard 内最终 artifacts。


### 4.1 `wikihow_computer_000002`: 艺术项目组织

- 标题：`how to be an organized artist 2`
- 规模：19 messages，8 tool calls，5 类工具
- 工具链：`open_resource -> list_records -> create_record -> list_records -> update_record -> create_record -> attach_resource -> list_records`
- 特点：典型 CRUD + final verification。先打开 Art Studio Organizer，检查项目是否存在，创建 Mural Project，读取 supplies，更新 white paint，再创建 study 并 attach 到项目。

### 4.2 `wikihow_computer_003813`: Minecraft 服务器更新

- 标题：`how to update a minecraft server`
- 规模：55 messages，26 tool calls，7 类工具
- 工具链核心：备份记录创建、旧文件删除、新 jar 下载、配置更新、服务启动、job polling、EULA 状态修改、最终 list 验证。
- 特点：当前最强的长程样本之一，包含文件状态变更、异步任务和多步依赖。

### 4.3 `wikihow_computer_000510`: 在线发布漫画

- 标题：`how to make a comic book online`
- 规模：37 messages，16 tool calls，10 类工具
- 工具链：`open_resource -> authenticate -> check_constraint -> locate_file -> create_upload_session -> upload_resource -> verify_resource -> update_record -> get_share_link -> send_message`
- 特点：覆盖登录、约束检查、文件定位、上传、发布、验证和通知，体现了上传发布类任务的可执行抽象。

### 4.4 `wikihow_computer_002621`: 澳大利亚签证申请

- 标题：`how to migrate to australia 3`
- 规模：39 messages，16 tool calls，3 个 user turn，10 类工具
- refinement patterns：`long_horizon_dependency`、`state_inspection`、`clarification`、`final_verification`
- 特点：先读取 application/checklist，再定位材料、上传、attach、提交并验证状态。Stage4 加入用户确认，使轨迹更接近真实高风险流程。

### 4.5 `wikihow_computer_003851`: Minecraft 论坛发帖

- 标题：`how to post a question on the minecraft forums`
- 规模：46 messages，13 tool calls，3 个 user turn，9 类工具
- refinement patterns：`long_horizon_dependency`、`error_recovery`、`final_verification`
- 特点：包含账号创建冲突、用户补充替代用户名/邮箱、发帖、上传截图、attach 附件和最终验证，适合展示错误恢复。

## 5. 早期 Gemini31Pro Stage3 对照样本

对照数据来自 `outputs/stage3/artifacts/gemini31pro_wikihow_computer5_stage3_artifacts.jsonl`。需要先说明：这个 artifact 文件实际只有 4 条，不是 5 条；对应 Stage3 请求有 5 条，其中 `wikihow_computer_000096` 在模型输出阶段失败，没有 materialize 成 artifact。

这批数据不是 no-validation 版本。它已经有 `environment`，并且现有 4 条都通过了 strict trajectory validation 和 execution validation。它和当前主 pipeline 的主要区别是：当时还没有统一使用 canonical atomic tool bank，工具仍然偏任务专属、偏高层；也没有 Stage4 refinement、tool-bank validation 和大规模 repair/fallback 机制。

对照数据整体统计：

| 指标 | Gemini31Pro 早期 Stage3 |
| --- | --- |
| Stage3 requests | 5 |
| materialized artifacts | 4 |
| failed output id | `wikihow_computer_000096` |
| environment rows | 4/4 |
| strict validation | 4/4 |
| execution validation | 4/4 |
| tool-bank validation | 未生成对应文件 |
| messages | min 19 / median 21 / max 23 |
| tool calls | min 5 / median 5.5 / max 6 |
| user turns | min 4 / median 4 / max 5 |

### 5.1 `wikihow_computer_000038`: DreamBox 登录

- 任务：使用 school code、classroom 和 student credentials 登录 DreamBox，并可选创建 parent account。
- 规模：19 messages，5 tool calls，4 个 user turns。
- 工具链：`validate_school_code -> validate_school_code -> list_classroom_students -> login_student -> create_parent_account`
- 优点：包含错误恢复，先处理无效 school code，再用正确 code 获取 classroom students，最后登录并创建家长账号。
- 问题：工具是任务专属工具，例如 `validate_school_code`、`login_student`、`create_parent_account`；`list_classroom_students` 的 tool response 还直接暴露学生 password，这在真实安全语境中不合理；没有独立的 `verify_session`。
- 对比当前：当前主 pipeline 会倾向于用 `open_resource/list_records/authenticate/verify_session` 这类可复用工具表达登录流程，并要求最终 verification。

### 5.2 `wikihow_computer_000068`: Ricoh 扫描并转发邮件

- 任务：登录 Ricoh MP C5503，配置扫描，扫描到邮箱，再转发给收件人。
- 规模：23 messages，6 tool calls，5 个 user turns。
- 工具链：`authenticate_scanner -> authenticate_scanner -> configure_scan -> start_scan -> logout_scanner -> forward_email`
- 优点：有错误恢复，先用错误密码登录失败，再由用户提供正确密码；扫描 subject 和 recipient 由用户明确提供。
- 问题：`configure_scan/start_scan/forward_email` 都是任务专属动作，`start_scan` 同时完成扫描和邮件生成，缺少当前 pipeline 中的 `start_job/poll_job/verify_job_result/list_records/get_record/create_record/attach_resource/update_record/verify_delivery` 等可观察中间状态。
- 对比当前：当前路线会把扫描、异步 job、邮件记录、附件绑定和投递验证拆成更原子的工具调用。

### 5.3 `wikihow_computer_000075`: 创建并导出 CSV

- 任务：创建 spreadsheet、写入 header 和一行数据，并导出 CSV。
- 规模：22 messages，6 tool calls，4 个 user turns。
- 工具链：`export_to_csv -> list_spreadsheets -> create_spreadsheet -> add_row -> add_row -> export_to_csv`
- 优点：先尝试导出用户给定的 `my_inventory`，失败后 list existing spreadsheets，再根据用户确认创建新表并导出，错误恢复比较自然。
- 问题：`create_spreadsheet/add_row/export_to_csv` 是 spreadsheet 专属工具；导出后只返回 `file_id`，没有后续 `verify_resource`；工具链较短，缺少资源打开、记录更新、导出验证等通用过程。
- 对比当前：当前对应能力更倾向于 `open_resource/create_record/update_record/export_resource/verify_resource`。

### 5.4 `wikihow_computer_000090`: BackCountry Navigator GPX 导出

- 任务：从 BackCountry Navigator Pro 的 trip database 中导出 GPX track 并分享。
- 规模：20 messages，5 tool calls，4 个 user turns。
- 工具链：`list_tracks -> list_trip_databases -> list_tracks -> export_track -> share_file`
- 优点：能处理用户给出的 `Weekend Trip` 数据库不存在的问题，随后列出 default database 并让用户确认 track。
- 问题：`export_track/share_file` 是 domain-specific 高层工具；没有 `open_resource`，也没有独立的 `verify_resource` 或 `get_share_link`；最终分享状态只由 `share_file` 一步返回。
- 对比当前：当前路线会把 app 打开、数据库 listing、track retrieval、export、verify 和 share link 拆成更标准的 tool-use 链。

## 6. 早期 Stage3 与当前主 Pipeline 的关键差异

| 维度 | Gemini31Pro 早期 Stage3 | 当前 execution-grounded pipeline |
| --- | --- | --- |
| 样本规模 | 5 requests / 4 artifacts | 3261 SFT records |
| 可执行环境 | 已有 `environment` | 已有 `environment`，并经过 repair/fallback 大规模流程 |
| 工具形态 | 任务专属高层工具，如 `login_student`、`start_scan`、`export_track` | canonical atomic tools，如 `open_resource`、`list_records`、`create_record`、`update_record`、`verify_resource` |
| tool response | 可通过 execution validation | 先 canonicalize，再 strict/execution/tool-bank 三重验证 |
| 轨迹长度 | 5-6 次 tool call，用户轮次较多 | median 7 次 tool call，p90 10，更多由工具发现状态 |
| 复杂化 | 无 Stage4 refinement | 有 Stage4 refinement，增加状态检查、长程依赖、错误恢复和最终验证 |
| 主要问题 | 工具不可复用、动作过粗、部分安全/隐私建模不自然 | DSL 环境仍是轻量模拟，不等价于真实 app/API |

一句话总结：这批早期 Gemini31Pro Stage3 数据已经证明了“环境可执行”路线可行，但还停留在任务专属工具阶段；当前主 pipeline 的进步主要在工具原子化、可复用 tool bank、最终验证、Stage4 refinement 和规模化生成。

## 7. 后续建议

1. **暂停期间先做质量审计**
   从当前 3261 条中按 domain、tool-call 长度、是否 Stage4 refinement、是否 repair、是否 fallback 分层抽样，人工检查 50-100 条。

2. **补全全局分析脚本**
   固化统计项：tool-call 数、user turn 数、error recovery 比例、final verification 比例、Stage4 fallback 比例、domain 分布、strict/execution/tool-bank 失败原因。

3. **继续优化 Stage1**
   重点诊断低通过率 shard，判断是输入分布变化，还是 prompt 对 computer-use 的定义过窄。

4. **保留当前路线，但不要过度追求真实 rollout**
   当前用户偏好是“模型先生成完整 response，再由 replay engine 验证和替换 response”。这比 step-by-step rollout 更稳定，也更适合大规模生成。

5. **保存早期 Gemini31Pro Stage3 作为 ablation baseline**
   这 4 条早期样本适合展示从“任务专属可执行工具”到“canonical atomic tools + Stage4 refinement + 大规模验证”的改进路径。
