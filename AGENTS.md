# Repository Guidelines

## Project Structure & Module Organization

This repository implements a GEM-style tool-use data synthesis pipeline. Python
entry points live in `scripts/`, prompt templates in `prompts/`, canonical tool
definitions in `config/tool_bank.json`, and small sample inputs in `data/`.
Generated artifacts belong under `outputs/` and are ignored by Git except for
`outputs/README.md`. `legacy_no_validation/` keeps the earlier no-execution
baseline for comparison.

## Build, Test, and Development Commands

Run commands from the repository root.

```bash
python3 scripts/toy_gem_pipeline.py
```

Runs the offline deterministic smoke pipeline and writes artifacts to
`outputs/toy/`.

```bash
python3 scripts/build_llm_requests.py --stage stage1 --input data/sample_texts.jsonl --output outputs/stage1/requests/stage1.jsonl
python3 scripts/validate_environment.py --input outputs/stage2/artifacts/stage2_artifacts.jsonl --output outputs/stage2/validation/environment.jsonl
python3 scripts/validate_tool_bank.py --input outputs/stage2/artifacts/stage2_artifacts.jsonl --output outputs/stage2/validation/tool_bank.jsonl --require-discoverable-record-ids
python3 scripts/validate_execution.py --input outputs/stage3/artifacts/stage3_trajectories.jsonl --output outputs/stage3/validation/execution.jsonl
python3 scripts/canonicalize_tool_responses.py --input outputs/stage3/artifacts/stage3_trajectories.jsonl --output outputs/stage3/artifacts/stage3_trajectories_canonical.jsonl
python3 scripts/build_llm_requests.py --stage stage4 --input outputs/stage3/artifacts/stage3_trajectories.jsonl --output outputs/stage4/requests/stage4_requests.jsonl
python3 scripts/quality_gate.py --input outputs/stage3/artifacts/stage3_trajectories.jsonl --trajectory-validation outputs/stage3/validation/trajectory_strict.jsonl --execution-validation outputs/stage3/validation/execution.jsonl --tool-bank-validation outputs/stage3/validation/tool_bank.jsonl
python3 scripts/run_pipeline.py --input data/wikihow_computer_100.jsonl --output-dir outputs/runs/wikihow_computer --candidate-limit 50 --target 10 --provider gemini --gemini-thinking-budget 0 --workers 4 --retries 1 --trajectory-repair-rounds 2
```

Use `execute_llm_requests.py` only after setting provider credentials in
environment variables such as `GEMINI_API_KEY` or
`GEM_LLM_BASE_URL`/`GEM_LLM_API_KEY`/`GEM_LLM_MODEL`. `rollout_stage3.py` is an
optional ablation path, not the default pipeline.
`execute_llm_requests.py --resume` reuses only rows whose request hash still
matches the current prompt/messages.
`run_pipeline.py` canonicalizes tool responses before validation by default;
pass `--no-canonicalize-tool-responses` only when inspecting raw model outputs.
If Stage 4 refinement invalidates a row that already passed Stage 3 validation,
the runner falls back to the Stage 3 row for final export.
For Gemini 2.5 Flash style thinking models, pass `--gemini-thinking-budget 0`
to keep JSON output complete and reduce latency.
Stage 4 refines messages only; after materializing refined outputs, re-run
strict trajectory, execution, and tool-bank validation.

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, type hints where useful, `argparse` CLIs,
`pathlib.Path`, and UTF-8 JSON/JSONL I/O. Keep scripts deterministic unless an
LLM endpoint is explicitly configured. Use `snake_case` for functions, files,
variables, and generated tool names.

## Testing Guidelines

There is no pytest suite yet. Treat `toy_gem_pipeline.py`, `py_compile`, and the
validators as the regression suite. Before changing prompts, schemas, or replay
logic, run the toy pipeline and relevant validators on representative artifacts.
If adding pytest coverage, place tests in `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines

Use short imperative commit messages, for example `Add replay validation gate`.
Pull requests should describe affected pipeline stages, commands run, changed
schemas/prompts, and any model or endpoint assumptions. Do not commit API keys,
large generated outputs, local logs, or private model responses.
