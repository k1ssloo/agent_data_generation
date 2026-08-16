# Output Artifact Layout

Artifacts are grouped by pipeline stage and by role in the pipeline.

- `toy/`: deterministic offline smoke-test outputs from `toy_gem_pipeline.py`, including `execution_validation.jsonl` when tool calls are replayed against executable environments.
- `stage1/requests/`: text-filtering request JSONL files.
- `stage1/model_outputs/`: raw OpenAI-compatible model responses for filtering.
- `stage2/requests/`: workflow/tool extraction request JSONL files.
- `stage2/model_outputs/`: raw model responses for workflow/tool extraction.
- `stage2/artifacts/`: materialized records containing source text, workflow, and tools.
- `stage2/validation/`: tool-schema and tool-atomicity lint results.
- `stage3/requests/`: trajectory-generation request JSONL files.
- `stage3/model_outputs/`: raw trajectory-generation model responses.
- `stage3/artifacts/`: materialized records containing source text, workflow, tools, and messages.
- `stage3/validation/`: trajectory schema and grounding validation results.
- `stage4/requests/`: refinement request JSONL files built from Stage 3 trajectories.
- `stage4/model_outputs/`: raw refinement model responses.
- `stage4/artifacts/`: materialized refined trajectories; Stage 4 replaces messages and preserves tools/environments.
- `stage4/validation/`: strict grounding, replay execution, and tool-bank validation for refined trajectories.
- `sft/`: OpenAI-message-style records ready for later conversion to a model-specific SFT template.
- `task_first/`: generated task-first request batches, materialized task bundles,
  hidden-environment rollout reports, execution traces, and causal validation
  reports. Reference plans are oracle proofs and must not be copied into policy
  training records.
