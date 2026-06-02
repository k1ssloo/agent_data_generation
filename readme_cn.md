# GEM 工具调用数据合成

本仓库是一个轻量级 GEM 风格复现与扩展，用于从过程性文本中合成多轮工具调用轨迹。当前默认路线是：Stage 1 过滤有效文本，Stage 2 抽取 workflow、工具和可执行环境，Stage 3 一次性生成完整对话轨迹，Stage 4 对轨迹做复杂化 refinement，然后用 replay engine 验证模型写出的 tool response 是否能由环境 DSL 执行得到。

## 核心思路

- **规范工具库**：Stage 2 优先从 `config/tool_bank.json` 选择原子工具，减少模型随意发明任务专属 API。
- **可执行环境**：每条 Stage 2 artifact 都包含 `environment.initial_state` 和 `tool_rules`，格式为 `text-exec-dsl-v0`。
- **一次性轨迹生成**：Stage 3 让模型直接生成 OpenAI messages 风格的完整多轮 tool-use 轨迹。
- **Stage 4 复杂化**：Stage 4 固定原来的 `workflow/tools/environment`，只重写 `messages`，增加长程依赖、状态检查、澄清、错误恢复、条件分支和最终验证。
- **Replay 验证**：`validate_execution.py` 会重放每个 tool call，并拒绝与 DSL 执行结果不一致的轨迹。
- **工具响应规范化**：`canonicalize_tool_responses.py` 可以用 DSL 重放结果替换模型写出的 tool message content，但不改变模型生成的对话和 tool-call 决策。

## 目录结构

```text
config/                    规范原子工具库。
data/                      小规模示例输入 JSONL。
prompts/                   各阶段 prompt、repair prompt 和 rollout prompt。
scripts/                   请求构建、模型调用、materialize、验证和导出脚本。
legacy_no_validation/      早期无执行验证路线，用于对比。
outputs/README.md          输出目录说明；生成文件默认被 Git 忽略。
TOOL_BANK_REFACTOR_PROCESS.md
```

大规模生成结果、模型日志、Python 缓存和本地密钥不应提交到 Git。API key 请放在环境变量中。

## 快速开始

运行离线 toy smoke pipeline：

```bash
python3 scripts/toy_gem_pipeline.py
```

构建并执行 Stage 1 请求：

```bash
python3 scripts/build_llm_requests.py \
  --stage stage1 \
  --input data/wikihow_computer_5.jsonl \
  --output outputs/stage1/requests/stage1_requests.jsonl

export GEMINI_API_KEY=<your-key>
export GEMINI_MODEL=gemini-3.5-flash

python3 scripts/execute_llm_requests.py \
  --provider gemini \
  --input outputs/stage1/requests/stage1_requests.jsonl \
  --output outputs/stage1/model_outputs/stage1_outputs.jsonl \
  --max-tokens 1024 \
  --temperature 0.1
```

将模型输出合并回 artifact：

```bash
python3 scripts/materialize_llm_outputs.py \
  --stage stage1 \
  --base data/wikihow_computer_5.jsonl \
  --llm-output outputs/stage1/model_outputs/stage1_outputs.jsonl \
  --output outputs/stage1/artifacts/stage1_filtered.jsonl
```

大规模生成时推荐直接使用编排脚本。它会串起 Stage 1 到 Stage 4，并在
Stage 2 和轨迹阶段自动运行验证、重试和 repair，最后只导出全部验证通过
的 SFT 数据：

```bash
python3 scripts/run_pipeline.py \
  --input data/wikihow_computer_100.jsonl \
  --output-dir outputs/runs/wikihow_computer \
  --candidate-limit 50 \
  --target 10 \
  --provider gemini \
  --gemini-thinking-budget 0 \
  --workers 4 \
  --retries 1 \
  --stage2-repair-rounds 1 \
  --trajectory-repair-rounds 2 \
  --repair-max-tokens 12288
```

如果手动分阶段运行，`execute_llm_requests.py` 现在支持 `--workers`、
`--retries`、`--resume`、`--checkpoint-every` 和
`--gemini-thinking-budget`，适合长批次断点续跑。对于 Gemini 2.5 Flash
这类 thinking 模型，`--gemini-thinking-budget 0` 可以降低延迟，并避免
thinking tokens 挤占 JSON 输出空间。
断点续跑会校验 request hash，prompt 或上游 artifact 变化后不会误用旧输出。
`run_pipeline.py --repair-max-tokens` 只提高 Stage 2/轨迹 repair 的 JSON
输出预算，不会放慢首轮生成。
`run_pipeline.py` 默认会在轨迹验证前规范化 tool response；如果需要检查
模型原始 tool output，可以加 `--no-canonicalize-tool-responses`。
如果 Stage 4 refinement 把原本已经验证通过的 Stage 3 轨迹改坏，runner
会自动回退到 Stage 3 的有效版本，避免 refinement 降低最终产出率。

## Stage 2：工具和环境

```bash
python3 scripts/build_llm_requests.py \
  --stage stage2 \
  --input outputs/stage1/artifacts/stage1_filtered.jsonl \
  --output outputs/stage2/requests/stage2_requests.jsonl
```

Stage 2 执行并 materialize 后，先做环境和工具库验证：

```bash
python3 scripts/validate_environment.py \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage2/validation/environment.jsonl

python3 scripts/validate_tool_bank.py \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage2/validation/tool_bank.jsonl \
  --require-discoverable-record-ids
```

如果 Stage 2 环境或工具可发现性失败，可以使用 `build_llm_requests.py --stage stage2_repair` 构建修复请求。

## Stage 3：轨迹生成

Stage 3 默认使用 one-shot 方式生成完整多轮对话，包括 assistant tool calls 和 tool responses：

```bash
python3 scripts/build_llm_requests.py \
  --stage stage3 \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage3/requests/stage3_requests.jsonl

python3 scripts/materialize_llm_outputs.py \
  --base outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --llm-output outputs/stage3/model_outputs/stage3_outputs.jsonl \
  --stage stage3 \
  --output outputs/stage3/artifacts/stage3_trajectories.jsonl
```

`rollout_stage3.py` 仍保留为可选消融路线。默认路线中，replay engine 只做验证，不负责生成轨迹。

## Stage 4：Refinement

Stage 4 接收 Stage 3 轨迹、workflow、工具、环境、轨迹统计和验证错误，要求模型在不修改工具和环境的前提下重写 `messages`：

```bash
python3 scripts/build_llm_requests.py \
  --stage stage4 \
  --input outputs/stage3/artifacts/stage3_trajectories.jsonl \
  --validation outputs/stage3/validation/trajectory_strict.jsonl \
  --execution-validation outputs/stage3/validation/execution.jsonl \
  --output outputs/stage4/requests/stage4_requests.jsonl

python3 scripts/materialize_llm_outputs.py \
  --base outputs/stage3/artifacts/stage3_trajectories.jsonl \
  --llm-output outputs/stage4/model_outputs/stage4_outputs.jsonl \
  --stage stage4 \
  --output outputs/stage4/artifacts/stage4_refined.jsonl
```

materializer 会保留 Stage 3 的 `workflow/tools/environment`，只替换 `messages`，并记录 `refinement_patterns`、`complexity_changes` 和 `stage4_complexity`。

## 验证和质量门

推荐对 refined 轨迹同时运行 strict grounding、execution replay 和 tool-bank 验证：

```bash
python3 scripts/validate_trajectories.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --output outputs/stage4/validation/trajectory_strict.jsonl \
  --strict-grounding \
  --require-workflow-tools \
  --min-tool-calls 6 \
  --max-user-turns 4 \
  --require-final-verification \
  --allow-control-arg-literals

python3 scripts/validate_execution.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --output outputs/stage4/validation/execution.jsonl

python3 scripts/quality_gate.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --trajectory-validation outputs/stage4/validation/trajectory_strict.jsonl \
  --execution-validation outputs/stage4/validation/execution.jsonl \
  --tool-bank-validation outputs/stage4/validation/tool_bank.jsonl \
  --output outputs/analysis/quality_gate.json
```

质量门不会修复数据，只负责汇总 completion、strict、execution 和最终可用率。
手动运行时，建议在 execution validation 前先规范化工具响应：

```bash
python3 scripts/canonicalize_tool_responses.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --output outputs/stage4/artifacts/stage4_refined_canonical.jsonl
```

## 导出 SFT

```bash
python3 scripts/convert_to_sft.py \
  --trajectories outputs/stage4/artifacts/stage4_refined.jsonl \
  --validation outputs/stage4/validation/trajectory_strict.jsonl \
  --extra-validation outputs/stage4/validation/execution.jsonl \
  --extra-validation outputs/stage4/validation/tool_bank.jsonl \
  --output outputs/sft/sft_openai_messages.jsonl
```

推荐数据分层：

- **Gold**：Stage 4 refined + strict grounding + execution + tool-bank 全部通过。
- **Repair pool**：Stage 3 或 Stage 4 中 grounding/replay 失败但可修复的样本。
- **Ablation**：step-by-step rollout 轨迹，使用同一套验证器评估。

## 本地 Qwen Teacher

可以用 vLLM 启动 OpenAI-compatible teacher：

```bash
PYTHONNOUSERSITE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
MODEL_PATH=/path/to/Qwen3-VL-32B-Instruct \
SERVED_MODEL_NAME=qwen32b-teacher \
PORT=18032 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.90 \
TENSOR_PARALLEL_SIZE=2 \
scripts/start_qwen_teacher_vllm.sh
```

然后通过 OpenAI-compatible 环境变量调用：

```bash
export GEM_LLM_BASE_URL=http://127.0.0.1:18032/v1
export GEM_LLM_API_KEY=local
export GEM_LLM_MODEL=qwen32b-teacher
python3 scripts/execute_llm_requests.py \
  --input outputs/stage1/requests/stage1_requests.jsonl \
  --output outputs/stage1/model_outputs/stage1_outputs.jsonl \
  --max-tokens 1024 \
  --temperature 0.1
```

## 注意事项

Replay engine 验证的是合成环境内的一致性，不代表真实外部软件执行成功。大规模生成时应把中间产物保存在 `outputs/`，检查 quality gate 报告，只提交源码、prompt、配置和必要的小型 fixture。
