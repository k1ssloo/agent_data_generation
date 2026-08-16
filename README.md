# GEM Tool-Use Data Synthesis

This repository is a compact reproduction and extension of a GEM-style pipeline
for synthesizing tool-use trajectories from procedural text. The default path is
one-shot trajectory generation, refinement, and replay validation: Stage 2
generates a lightweight environment DSL, Stage 3 generates full tool-use
dialogues, Stage 4 rewrites them for higher complexity, and the replay engine
checks whether model-written tool responses are executable.

The repository now also contains an additive task-first MVP based on
`TASK_FIRST_REFACTOR_PLAN.md`. It synthesizes executable task bundles, keeps the
reference plan and hidden state out of policy context, executes episode-specific
tools against a deterministic causal runtime, and validates runtime provenance
instead of trusting model-written causal labels. The original Stage 1--4 path is
kept as a baseline and remains unchanged.

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

Run the task-first offline fixture and causal quality gate:

```bash
python3 scripts/run_task_first_fixture.py \
  --bundle tests/fixtures/release_task \
  --output-dir outputs/task_first/release_fixture
```

This writes `episode.json`, `validation.json`, and a combined `result.json`.
Use the combined file when exporting training data.

Run the stronger offline gate and export only accepted public trajectories:

```bash
python3 scripts/evaluate_task_first.py \
  --bundle tests/fixtures/release_task \
  --output outputs/task_first/release_fixture/evaluation.json

python3 scripts/export_task_first_sft.py \
  --input outputs/task_first/release_fixture/result.json \
  --output outputs/task_first/release_fixture/sft.jsonl
```

Generate a small deterministic batch with one canonical API and several
episode-specific API renderings:

```bash
python3 scripts/synthesize_task_first_samples.py \
  --bundle tests/fixtures/release_task \
  --output-dir outputs/task_first/demo_samples \
  --count 4
```

The accepted OpenAI-message records are written to
`outputs/task_first/demo_samples/accepted_sft.jsonl`; full private execution and
validation reports remain under `results/` for auditing only.

Recursively evolve the task itself through registered semantic operators:

```bash
python3 scripts/evolve_task_bundles.py \
  --parent tests/fixtures/release_task \
  --output-dir outputs/task_first/evolution_strict_demo \
  --operators policy_freshness_coupling_v1 capacity_reservation_branch_v1
```

Each generation is admitted to the append-only archive only when its oracle
passes causal validation, the parent plan no longer solves it unchanged, at
least one semantic complexity dimension increases, and every declared
counterfactual requires a successful adapted strategy. Registered operators
cover fixture-specific policy/capacity changes and portable target audit,
execution-route, and asynchronous-readiness changes. Contract patches and
complete gate reports are saved under `audits/`; accepted child bundles are
saved under `bundles/`.

Run the complete WikiHow task factory with a configured Codex/OpenAI model:

```bash
python3 scripts/run_wikihow_task_factory.py \
  --input data/wikihow_computer_100.jsonl \
  --output-dir outputs/task_first/wikihow_factory \
  --config ../config.toml \
  --task-id wikihow_computer_000092 \
  --repair-rounds 3 \
  --recursive-generations 2 \
  --resume
```

The runner compiles verbatim-cited source evidence, generates a contract and
executable environment/reference solution together, repairs from deterministic
gate feedback, infers a portable semantic commit hook, and recursively searches
validated children. Cached seeds bind both `source_id` and `source_sha256`;
resume rejects a colliding ID with different text. Use `--regenerate-seed` only
when intentionally replacing source grounding.

For higher throughput on an OpenAI-compatible Responses endpoint, use the same
provider TOML with concurrent independent roots:

```bash
python3 scripts/run_wikihow_task_factory.py \
  --input data/wikihow_computer_100.jsonl \
  --output-dir outputs/task_first/wikihow_factory_fast \
  --config ../config.toml \
  --provider responses \
  --workers 2 \
  --bundle-candidates 2 \
  --patch-repair \
  --patch-reasoning-effort medium \
  --repair-rounds 3 \
  --recursive-generations 2 \
  --resume
```

Concurrency and multi-candidate sampling are opt-in; guarded patch repair is
enabled by default and can be disabled with `--no-patch-repair`. Every generated
or patched candidate still runs
the same static, execution, provenance, alternate-rendering, ablation, and
counterfactual gates. Patch repair is accepted only when the complete gate is
fully valid or its error set is a strict subset of the previous error set;
otherwise the factory falls back to a
complete bundle repair. A narrow deterministic fast path also repairs a final
domain observation that is missing goal evidence, but only when reference
execution already satisfies the goal and this is the sole error; the repaired
bundle must pass the same complete gate. A valid contract is not regenerated on every bundle
repair unless `--always-repair-contract` is set. The summary reports model-call
latency, per-stage request/character profiles, accepted roots per hour, and
accepted semantic episodes per hour (roots plus recursive descendants).
Cold-start and resume-only runs should be compared separately. Responses API output budgets
are expanded because reasoning tokens and JSON output share one limit.

An approved structurally valid contract is treated as an immutable task
specification during normal bundle repair. Missing state paths or initial-state
mismatches are owned by the candidate bundle and cannot be "fixed" by weakening
or rewriting the contract. `--always-repair-contract` exists only for legacy
experiments. A lower `--patch-reasoning-effort` is guarded by the complete gate
and same-round full-repair fallback; seed, contract, and full-bundle reasoning
remain at the configured quality baseline unless explicitly overridden. Patch
repair defaults to `medium` reasoning and receives the immutable contract plus
the rejected executable candidate, without a redundant copy of the source seed.

Successful JSON model stages are cached by exact prompt and non-secret provider
configuration under `outputs/.llm_cache`. Use `--cache-namespace experiment_2`
to request an independent sample family or `--no-llm-cache` to disable reuse.
If a root fails the complete quality gate, its bundle-generation cache entries
are evicted so the next run can resample the candidate; independently validated
seed and contract stages remain reusable. The cache never bypasses validation.

Contract repair receives the complete initial state plus capability, public-tool,
main-action, and counterfactual topology instead of a second copy of the full
executable bundle. Generated contracts are normalized to contract-only fields;
unknown executable payloads such as `environment`, `bindings`, or
`reference_plan` are rejected. The final candidate is still evaluated in full.

Use the default Codex command when establishing a new quality baseline. Enable
stage-specific reasoning or additional workers only after comparing acceptance
rate and causal metrics on a representative shard. Codex and HTTP providers can
process independent roots concurrently; begin with a small worker count that
fits the endpoint's rate and connection limits.

`--bundle-candidates N` samples independent executable bundles concurrently for
each already validated seed/contract and selects the strongest result using the
same full quality report. Invalid samples are not rendered as extra data and
are evicted from the model cache. Maximum generation concurrency is roughly
`workers * bundle_candidates`; size both values against the endpoint quota.

When optimizing an existing accepted corpus, pass its root directory with
`--quality-baseline-dir`. Each new task must meet or exceed the matching
baseline's steps, delayed-handle distance, handle-chain depth, semantic recovery
count, within-rollout observation branches, valid counterfactual count,
decision entropy, and necessary-action ratio. These dimensions are checked
component by component, so a higher score on one cannot hide regression on
another. Missing or currently invalid baselines stop the run.
With `--resume`, existing roots under the selected `--roots-subdir` are used as
automatic per-task baselines; tasks without an existing root are treated as new.
If the best candidate is below a baseline, the factory requests a fresh
independent quality sample only when `--quality-resample-candidates` is set to a
positive value. This experimental path is disabled by default: the tested
`xhigh` resample added 79.5 seconds and produced a weaker five-step design.
Generic rewriting of a weaker parent likewise requires the explicit
`--repair-quality-regressions` flag. The default is therefore a fast,
quality-preserving rejection.

The recursive search currently includes target-specific audit evidence,
environment-dependent route reservation, stateful async readiness, and a
portable semantic-failure recovery operator. Recursive instructions state only
outcome constraints. Tool discovery, failure diagnosis, repair evidence, and
route choice remain environment-discoverable through public schemas and
observations instead of being written as a workflow in the user prompt.

The recovery operator makes the same consequential action fail once with a
public error report, binds diagnosis and repair to the original target, and
requires the repaired action to be retried with fresh evidence. The async
operator reaches `pending` and `ready` branches in one rollout and makes the
later commit consume the readiness handle.

The fixture exercises a 15-action release task with hidden state, delayed handle
use, a failed quality run followed by semantic recovery, asynchronous run
handles, derived artifacts, and final observations covering every goal
predicate. The command exits nonzero when outcome or causal validation fails.

Run the dependency-free task-first tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Agent SFT And RL

The task-first runtime has a direct rLLM integration for supervised tool-call
training and online agentic RL. It exports public expert conversations for SFT
and separate bundle-backed tasks for fresh hidden-environment rollouts; private
initial state, contracts, oracle plans, and verifier predicates are never put in
policy context. See [`training/README.md`](training/README.md) for framework
selection, batch generation, export, and training commands.

Build task-first model requests in two stages:

```bash
python3 scripts/build_task_first_requests.py \
  --stage contract \
  --input path/to/executable_seed_summaries.jsonl \
  --output outputs/task_first/requests/contracts.jsonl

python3 scripts/build_task_first_requests.py \
  --stage bundle \
  --input path/to/approved_contracts.jsonl \
  --output outputs/task_first/requests/bundles.jsonl
```

Use `execute_llm_requests.py` for those request files, then materialize successful
contract and bundle outputs:

```bash
python3 scripts/materialize_task_first_outputs.py \
  --contracts outputs/task_first/model_outputs/contracts.jsonl \
  --bundles outputs/task_first/model_outputs/bundles.jsonl \
  --output-dir outputs/task_first/bundles
```

The request executor can also use the locally installed Codex CLI as a GPT
generation backend. An explicit Codex TOML is supported without copying its
credential into commands or output artifacts:

```bash
export GEM_CODEX_CONFIG=/path/to/config.toml
python3 scripts/execute_llm_requests.py \
  --provider codex \
  --workers 1 \
  --input outputs/task_first/requests/contracts.jsonl \
  --output outputs/task_first/model_outputs/contracts.jsonl
```

The bridge resolves the nearest matching `[projects."..."]` entry for the
current repository, injects a provider credential into the child process only,
runs an ephemeral read-only session, and disables unrelated MCP servers by
default. `GEM_CODEX_MODEL`, `GEM_CODEX_PROVIDER`, and
`GEM_CODEX_REASONING_EFFORT` override TOML selections. Set
`GEM_CODEX_DISABLE_MCP=0` only when generation genuinely needs an MCP server.
Do not commit a TOML that contains an API key; environment-only credentials are
still preferred.

Procedural WikiHow rows are semantic seed candidates, not executable seeds by
themselves. Promote a row only after compiling its workflow and replayed
solution into a task bundle, passing the oracle and causal gates, and recording
which later requirements are synthetic extensions rather than source claims.
Run the feasibility probe on existing WikiHow trajectories with:

```bash
python3 scripts/validate_wikihow_task_recursion.py \
  --source data/wikihow_computer_100.jsonl \
  --trajectories outputs/runs/query_grounding_wikihow5/stage4/repair_round1/canonicalized.jsonl \
  --output-dir outputs/task_first/wikihow_feasibility
```

The probe audits source affordances, compiles replayable workflows into hidden
causal environments, validates their reference solutions, and applies one
solution-first recursive rewrite. See `WIKIHOW_RECURSIVE_FEASIBILITY.md` for the
current evidence and limitations.

After full bundle validation, select distinct source tasks and export canonical
OpenAI-message trajectories with a fresh strict validation pass:

```bash
python3 scripts/export_validated_wikihow_corpus.py \
  --validation outputs/task_first/wikihow_probe/children_validation.jsonl \
  --output-dir outputs/task_first/wikihow_export \
  --count 20
```

The exporter writes `openai_messages.jsonl`, one rollout audit per semantic
episode, and `summary.json`. It does not count API renderings as distinct tasks.
This fast route amortizes an existing model-generated WikiHow workflow; it does
not remove the cold semantic-synthesis cost for previously unseen raw text.
Export metadata marks this validation as
`declared_executable_contract_only`: contract evidence coverage must not be
reported as natural-language instruction coverage.

Audit a selected corpus for individually and jointly removable workflow steps:

```bash
python3 scripts/audit_task_corpus_quality.py \
  --input-dir outputs/task_first/wikihow_probe/children \
  --ids-from outputs/task_first/wikihow_export/openai_messages.jsonl \
  --output outputs/task_first/wikihow_export/semantic_quality_audit.jsonl \
  --summary outputs/task_first/wikihow_export/semantic_quality_summary.json
```

The audit reports three distinct levels: executable-contract validity, absence
of removable domain mutations, and an irreducible strict workflow. Adaptive
validity additionally requires a validated planning alternative or semantic
recovery. Instruction-to-contract coverage and domain-state consistency remain
explicit semantic-audit requirements; they are not inferred from path coverage.

New WikiHow extracts use the original dataset `source_index` in row IDs so IDs
remain stable across target sizes and filters. `--legacy-sequential-ids` exists
only to reproduce older files whose selected-row ordinals can collide.

Assemble several validated SFT shards into one auditable corpus:

```bash
python3 scripts/assemble_task_first_corpus.py \
  --input outputs/task_first/shard_a/accepted_sft.jsonl \
          outputs/task_first/shard_b/accepted_sft.jsonl \
  --output outputs/task_first/corpus/accepted_sft.jsonl \
  --report outputs/task_first/corpus/report.json
```

The assembler rejects malformed or duplicate OpenAI-message trajectories and
reports source, operator, recursion, API-rendering, and causal-complexity
coverage. Public SFT metadata retains source hashes and lineage labels but not
hidden state, contracts, or reference plans. Corpus size is reported separately
as `unique_source_tasks`, `unique_semantic_episodes`, `recursive_descendants`,
and `rendered_training_rows`; opaque API renderings are not counted as new
semantic tasks. Alternate APIs use opaque callable names but retain explicit
semantic affordances in tool and parameter descriptions. The identifiability
gate rejects missing descriptions and indistinguishable description/schema
groups before export.

Run a model policy step by step without exposing the contract, hidden state, or
reference plan:

```bash
python3 scripts/rollout_task_episode.py \
  --bundle tests/fixtures/release_task \
  --output outputs/task_first/release_rollout.json \
  --provider gemini
```

Task-first modules are grouped under `task_factory/`, `runtime/`, `rollout/`,
`causal_validation/`, and `schemas/`. This MVP implements the first vertical
slice of the refactor plan; large-scale recursive reseeding, sandboxed host
workspaces, model-generated evolution patches, and general
alternative-plan search remain later phases. Deterministic recursive evolution,
MAP-Elites-style selection, and counterfactual intervention gates are
implemented.
The current causal gate classifies every tool argument as `user_grounded`,
`tool_observation_grounded`, `schema_grounded`, `agent_choice`, `derived`, or
`unexplained`. Sensitive literals cannot be justified by a schema enum. The
gate also checks delayed handles, derived chains, semantic recovery,
goal-grounded verification, action ablation, and deterministic API rerendering.
Counterfactual axes must be observed before the first strategy divergence.
Final goal evidence must be returned by one read-only domain observation after
the last goal-state mutation. Reads accumulated earlier in the trace, or added
to the consequential mutation itself, cannot satisfy this gate.
`state_dependent_transition_count` records changing runtime branches, while
`decision_entropy_bits` credits only valid grounded counterfactual policies;
deterministic `pending -> ready` polling does not create strategy entropy.

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
  --trajectory-repair-rounds 2 \
  --repair-max-tokens 12288
```

`execute_llm_requests.py` also supports `--workers`, `--retries`, `--resume`,
`--checkpoint-every`, and `--gemini-thinking-budget` for manual staged runs.
Resume is request-hash aware, so stale outputs are not reused after prompt or
upstream artifact changes.
`run_pipeline.py --repair-max-tokens` gives Stage 2/trajectory repair calls a
larger JSON budget without slowing the first-pass generations.
For Gemini 2.5 Flash style thinking models, `--gemini-thinking-budget 0`
reduces latency and prevents thinking tokens from crowding out JSON output.

`run_pipeline.py` canonicalizes tool responses before trajectory validation by
default. Use `--no-canonicalize-tool-responses` when you need to inspect raw
model-written tool outputs. If Stage 4 refinement makes a previously valid
Stage 3 trajectory fail, the runner falls back to the valid Stage 3 version so
yield is not reduced by refinement.

For 10k-scale WikiHow runs, first build a larger input file, then execute the
pipeline in resumable shards:

```bash
python3 scripts/extract_wikihow_computer_use.py \
  --output data/wikihow_computer_10000.jsonl \
  --target 10000 \
  --scan-limit 1000000 \
  --max-text-chars 6000 \
  --progress-every 10000

python3 scripts/run_pipeline_shards.py \
  --input data/wikihow_computer_10000.jsonl \
  --input-limit 10000 \
  --shard-size 100 \
  --output-dir outputs/runs/wikihow_computer_10k \
  --provider gemini \
  --gemini-thinking-budget 0 \
  --workers 4 \
  --retries 1 \
  --stage2-repair-rounds 1 \
  --trajectory-repair-rounds 2 \
  --repair-max-tokens 12288 \
  --min-shard-final-valid 1 \
  --continue-on-error
```

The sharded runner writes per-shard artifacts under `outputs/runs/.../shards/`,
aggregates passing SFT records into `sft_openai_messages.jsonl`, and writes
`shard_summary.json`. Re-running the same command skips shards with an existing
`summary.json` unless `--force` is supplied. `--min-shard-final-valid` is a
guard against silent zero-yield shards when provider credentials, quota, or
model behavior changes unexpectedly.

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
