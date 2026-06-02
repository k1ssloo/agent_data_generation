# GEM Tool-Use Data Synthesis

This repository is a compact reproduction and extension of a GEM-style pipeline
for synthesizing tool-use trajectories from procedural text. The default path is
one-shot trajectory generation, refinement, and replay validation: Stage 2
generates a lightweight environment DSL, Stage 3 generates full tool-use
dialogues, Stage 4 rewrites them for higher complexity, and the replay engine
checks whether model-written tool responses are executable.

## Key Ideas

- **Canonical tool bank**: Stage 2 selects reusable atomic tools from
  `config/tool_bank.json` instead of inventing task-specific APIs.
- **Executable environments**: each Stage 2 artifact includes
  `environment.initial_state` and per-tool `tool_rules` in `text-exec-dsl-v0`.
- **Stage 4 refinement**: the model rewrites Stage 3 messages to add long-range
  dependencies, state inspection, clarification, recovery, and final checks.
- **Replay validation**: `validate_execution.py` replays every tool call and
  rejects trajectories whose tool responses do not match the DSL.
- **Tool-response canonicalization**: `canonicalize_tool_responses.py` can
  replace model-written tool message contents with replayed DSL outputs while
  preserving the model-generated dialogue and tool-call decisions.
- **Optional rollout**: `rollout_stage3.py` can run step-by-step executable
  generation for ablations, but it is not the default data path.
- **Repair hooks**: Stage 2 and Stage 3 repair prompts can regenerate artifacts
  that fail environment, grounding, or execution checks.

## Repository Layout

```text
config/                    Canonical atomic tool bank.
data/                      Small sample input JSONL files.
prompts/                   Stage prompts, repair prompts, and rollout prompts.
scripts/                   Pipeline builders, executors, validators, optional rollout.
legacy_no_validation/      Prior no-execution baseline for comparison.
outputs/README.md          Output directory description; generated files ignored.
TOOL_BANK_REFACTOR_PROCESS.md
```

Generated outputs, model logs, Python caches, and large regenerated datasets are
ignored by Git. Keep API keys in environment variables, not files.

## Quick Start

Run the deterministic offline smoke pipeline:

```bash
python3 scripts/toy_gem_pipeline.py
```

Build and execute Stage 1 requests with Gemini:

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

Materialize Stage 1 outputs:

```bash
python3 scripts/materialize_llm_outputs.py \
  --stage stage1 \
  --base data/wikihow_computer_5.jsonl \
  --llm-output outputs/stage1/model_outputs/stage1_outputs.jsonl \
  --output outputs/stage1/artifacts/stage1_filtered.jsonl
```

For larger batches, use the orchestrated pipeline. It runs Stage 1 through
Stage 4, validates each gate, retries failed LLM calls, repairs failed Stage 2
and trajectory artifacts, and exports SFT records that pass all validators:

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
  --trajectory-repair-rounds 2
```

`execute_llm_requests.py` also supports `--workers`, `--retries`, `--resume`,
`--checkpoint-every`, and `--gemini-thinking-budget` for manual staged runs.
For Gemini 2.5 Flash style thinking models, `--gemini-thinking-budget 0`
reduces latency and prevents thinking tokens from crowding out JSON output.

`run_pipeline.py` canonicalizes tool responses before trajectory validation by
default. Use `--no-canonicalize-tool-responses` when you need to inspect raw
model-written tool outputs. If Stage 4 refinement makes a previously valid
Stage 3 trajectory fail, the runner falls back to the valid Stage 3 version so
yield is not reduced by refinement.

## Stage 2: Tool Bank and Environment

Build Stage 2 requests. The canonical tool bank is injected by default:

```bash
python3 scripts/build_llm_requests.py \
  --stage stage2 \
  --input outputs/stage1/artifacts/stage1_filtered.jsonl \
  --output outputs/stage2/requests/stage2_requests.jsonl
```

After executing and materializing Stage 2, validate the artifacts:

```bash
python3 scripts/validate_environment.py \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage2/validation/environment.jsonl

python3 scripts/validate_tool_bank.py \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage2/validation/tool_bank.jsonl \
  --require-discoverable-record-ids
```

For optional executable rollout experiments, run the readiness gate:

```bash
python3 scripts/validate_rollout_readiness.py \
  --input outputs/stage2/artifacts/stage2_artifacts.jsonl \
  --output outputs/stage2/validation/rollout_readiness.jsonl \
  --warnings-as-errors
```

Readiness failures matter for step-by-step rollout. For the default one-shot
path, environment and tool-bank validation are the main Stage 2 gates.

## Stage 3: One-Shot Trajectories

Stage 3 asks the model to generate the initial full dialogue, including
assistant tool calls and tool responses:

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

`rollout_stage3.py` remains available as an optional ablation. In the default
path, the replay engine is a verifier: it does not produce the trajectory, it
checks whether the model-generated tool responses match the executable DSL.

## Stage 4: Refinement

Stage 4 rewrites the Stage 3 message sequence to increase complexity while
keeping the Stage 2 tools and executable environment fixed. It receives the
original trajectory, workflow, environment, trajectory statistics, and optional
validation errors:

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

The materializer preserves `workflow`, `tools`, and `environment` from Stage 3.
It only replaces `messages` and records `refinement_patterns`,
`complexity_changes`, and `stage4_complexity` deltas.

## Validation and Quality Gate

Validate refined trajectories. If Stage 4 is skipped, use the Stage 3 artifact
path in the same commands.

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

python3 scripts/validate_tool_bank.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --output outputs/stage4/validation/tool_bank.jsonl \
  --require-discoverable-record-ids
```

When running stages manually, canonicalize tool responses before strict
execution validation to remove response-format drift while keeping the generated
assistant tool calls intact:

```bash
python3 scripts/canonicalize_tool_responses.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --output outputs/stage4/artifacts/stage4_refined_canonical.jsonl
```

Summarize batch quality and fail fast if yield is too low:

```bash
python3 scripts/quality_gate.py \
  --input outputs/stage4/artifacts/stage4_refined.jsonl \
  --trajectory-validation outputs/stage4/validation/trajectory_strict.jsonl \
  --execution-validation outputs/stage4/validation/execution.jsonl \
  --tool-bank-validation outputs/stage4/validation/tool_bank.jsonl \
  --output outputs/analysis/quality_gate.json
```

## Convert to SFT

Export only records that pass all selected validators:

```bash
python3 scripts/convert_to_sft.py \
  --trajectories outputs/stage4/artifacts/stage4_refined.jsonl \
  --validation outputs/stage4/validation/trajectory_strict.jsonl \
  --extra-validation outputs/stage4/validation/execution.jsonl \
  --extra-validation outputs/stage4/validation/tool_bank.jsonl \
  --output outputs/sft/sft_openai_messages.jsonl
```

Recommended data tiers:

- **Gold**: refined trajectory + strict grounding + execution + tool-bank.
- **Repair pool**: Stage 3 trajectories that fail grounding or replay checks.
- **Ablation**: step-by-step rollout + the same validators.

## Local Qwen Teacher

An OpenAI-compatible local teacher can be served with vLLM:

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

Then call it with:

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

## Notes

The replay engine validates consistency with the synthetic environment, not with
real external software. Large-scale runs should keep generated artifacts in
`outputs/`, inspect quality-gate reports, and commit only source code, prompts,
configuration, and curated fixtures.
