# Legacy No-Validation Pipeline

This folder preserves the earlier GEM reproduction route that produced:

`gem_repro/outputs/sft/qwen32b_sft_openai_messages_smoke.jsonl`

It is intentionally separate from the current executable route. The current
main pipeline should continue to generate trajectories and immediately validate
them with tool/schema/environment/execution checks. This legacy folder keeps the
old baseline where Stage 3 can synthesize plausible tool observations without a
replayable environment.

## What This Version Does

1. Stage 2 asks the LLM to extract a workflow and OpenAI-style function tools.
   It does not ask for `environment`, `state_schema`, DSL actions, or effects.
2. Stage 3 asks the LLM to generate a multi-turn tool-use dialogue from the
   workflow and tools. Tool results are free-form synthetic observations.
3. SFT conversion writes OpenAI-message JSONL records directly from Stage 3
   trajectories. Validation is optional and disabled by default.

This matches the old smoke output style, including composite tools such as
`retry_print_with_alternative_printer_if_failed`.

## Reproduce The Smoke SFT Fixture

From the repository root:

```bash
python3 gem_repro/legacy_no_validation/scripts/reproduce_smoke_fixture.py
```

This writes:

`gem_repro/legacy_no_validation/outputs/qwen32b_sft_openai_messages_smoke.jsonl`

The script bootstraps a legacy Stage 3 fixture from the historical smoke SFT
file, then runs the no-validation converter.

## Build Legacy LLM Requests

```bash
python3 gem_repro/legacy_no_validation/scripts/build_legacy_requests.py \
  --stage stage2 \
  --input gem_repro/data/sample_texts.jsonl \
  --output gem_repro/legacy_no_validation/outputs/requests/stage2_requests.jsonl

python3 gem_repro/legacy_no_validation/scripts/build_legacy_requests.py \
  --stage stage3 \
  --input gem_repro/outputs/stage2/artifacts/qwen32b_stage2_artifacts_smoke.jsonl \
  --output gem_repro/legacy_no_validation/outputs/requests/stage3_requests.jsonl
```

Submit these requests with the existing OpenAI-compatible executor if needed.
For larger batches, the legacy folder also provides a bounded concurrent
executor:

```bash
GEM_LLM_BASE_URL=http://127.0.0.1:18009/v1 \
GEM_LLM_API_KEY=EMPTY \
GEM_LLM_MODEL=qwen3vl-32b \
python3 gem_repro/legacy_no_validation/scripts/execute_legacy_requests_parallel.py \
  --input gem_repro/legacy_no_validation/outputs/wikihow100_run/stage2_requests.jsonl \
  --output gem_repro/legacy_no_validation/outputs/wikihow100_run/stage2_outputs.jsonl \
  --workers 4
```

Merge raw model outputs back into legacy artifacts with:

```bash
python3 gem_repro/legacy_no_validation/scripts/materialize_legacy_outputs.py \
  --stage stage2 \
  --base gem_repro/data/sample_texts.jsonl \
  --llm-output gem_repro/legacy_no_validation/outputs/llm/stage2_outputs.jsonl \
  --output gem_repro/legacy_no_validation/outputs/artifacts/stage2_artifacts.jsonl
```

## Convert Without Validation

```bash
python3 gem_repro/legacy_no_validation/scripts/convert_legacy_to_sft.py \
  --trajectories gem_repro/legacy_no_validation/fixtures/qwen32b_stage3_smoke_legacy.jsonl \
  --output gem_repro/legacy_no_validation/outputs/qwen32b_sft_openai_messages_smoke.jsonl
```

Use `--validation path/to/validation.jsonl` only if you explicitly want the old
schema filter. Do not use this folder for the executable text-to-trajectory
pipeline.
