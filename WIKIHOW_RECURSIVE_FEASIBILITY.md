# WikiHow-to-Environment Recursive Feasibility

## Conclusion

WikiHow is feasible as a **semantic seed source**, but not as a directly trusted
executable task source. A useful row supplies the normal workflow, user-visible
goal, objects, and some ordering constraints. It usually does not supply a
complete hidden state machine, precise failure semantics, counterfactual
variants, or an outcome verifier.

The viable construction is therefore:

```text
WikiHow text
  -> workflow and object extraction
  -> executable environment + reference solution compilation
  -> oracle/outcome/causal validation
  -> solution-first recursive operator
  -> aligned environment, solution, contract, and instruction
  -> old-solution and counterfactual rejection
  -> diversity-capped verified seed pool
```

This follows the central ordering in RST: grow executable behavior first, then
align environment, verifier/contract, and public instruction. Recursive
requirements are marked as synthetic extensions and are never attributed to
the WikiHow article.

## What WikiHow Provides

- A recognizable user goal and normal-case procedure.
- Candidate entities, resources, settings, and artifacts.
- Partial precedence and data-flow constraints.
- Occasional alternatives, error handling, asynchronous work, and observable
  completion cues.

It does **not** reliably provide:

- Complete preconditions and side effects for every action.
- Stable identifiers or schemas for tool outputs.
- Hidden resource state, timing, retries, or failure distributions.
- Counterfactual environments with different valid strategies.
- A semantic outcome verifier or anti-shortcut rules.

Those missing elements may be generated only as explicit, discoverable, and
executable transformation contracts.

## Implemented Probe

`task_factory/wikihow_compiler.py` lifts an already replayable Stage 2/3
WikiHow artifact into `causal-runtime-v1`:

1. Replays the legacy environment from its initial state.
2. Uses the observed calls as a private solvability witness.
3. Derives state effects from replayed before/after state, rather than trusting
   model-written causal labels.
4. Marks only values actually returned by prior tools as required provenance.
5. Builds outcome predicates over changed state and makes the final inspection
   expose those state paths.
6. Rejects rows with no observable state change, mismatched replay, missing
   public schemas, or state that cannot be represented without guessing.

`scripts/validate_wikihow_task_recursion.py` then applies the complete gate:

- bundle static validation;
- oracle execution from a fresh hidden state;
- goal evidence and provenance validation;
- delayed-handle and chain-depth measurement;
- action ablation;
- recursive child validation;
- unchanged-parent-solution rejection.

Recursive acceptance in `task_factory/evolve.py` now also requires at least
60% of reference actions to be necessary. This prevents complexity from being
inflated by inert calls.

## Empirical Results

The source-level audit over `data/wikihow_computer_100.jsonl` found:

| Signal | Rows |
| --- | ---: |
| Observation language | 94 / 100 |
| Mutation language | 95 / 100 |
| Branch/alternative language | 90 / 100 |
| Explicit failure language | 22 / 100 |
| Async/progress language | 42 / 100 |
| Verification cues | 64 / 100 |
| Concrete artifact language | 82 / 100 |

The median extracted summary length was eight steps. A deliberately permissive
heuristic marked 51 rows as core-compilable candidates and 50 as having at
least one recursive affordance. These counts are candidate-recall estimates,
not acceptance rates.

Executable probes produced the following:

| Cohort | Attempted | Compiled | Base-valid | R1 accepted |
| --- | ---: | ---: | ---: | ---: |
| Earlier, unrefined WikiHow trajectories | 10 | 7 | 4 | 0 |
| Strictly repaired WikiHow trajectories | 5 | 2 | 2 | 1 |

The gap is informative: recursion does not repair an invalid seed. It must
start from a verified seed pool, as in RST.

The accepted PowerTeacher lineage demonstrates two executable rounds:

| Version | Steps | Max delayed use | Chain depth | Goal evidence | Necessary actions | Counterfactuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Compiled WikiHow root | 9 | 2 | 3 | 100% | 100% | 0 |
| R1 audit checkpoint | 11 | 4 | 4 | 100% | 90.9% | 0 |
| R2 execution route | 13 | 6 | 4 | 100% | 92.3% | 1 |

For both R1 and R2, the unchanged parent solution fails on the child. In R2,
the baseline environment requires a fallback-route reservation, while the
counterfactual environment requires a primary-route reservation. The adapted
solution passes and the stale strategy fails.

Reproducible artifacts are under:

- `outputs/task_first/wikihow_feasibility_v3/`
- `outputs/task_first/wikihow_recursive_r2_probe/`

### Model-generated task factory validation

The current task-first runner was also tested end to end with the model from
the provided TOML configuration. Unlike the earlier compiler probe, these roots
were generated from WikiHow text as a grounded seed, contract, hidden runtime,
public tools, and reference solution, then repaired only through deterministic
validation feedback.

| Source task | Root operator | Root steps | Delayed handle | Chain | Necessary actions | Accepted adaptive endpoint |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Ricoh scan-to-email and forward | artifact provenance | 21 | 8 | 5 | 71.43% | 26 steps, failure recovery + route |
| Windows offline map download | alternative plan | 12 | 10 | 5 | 91.67% | 17 steps, failure recovery + route |

Every accepted root and child has 100% goal-evidence coverage, no unexplained
argument, no invariant violation, and remains valid after public API renaming.
Argument provenance covers ordinary string literals as well as handles; the
repaired scanner discovers its icon, account identity, opaque session, scan
mode, and mailbox identity through natural interaction surfaces. At every
recursive generation the unchanged parent plan fails on the child. Route
children additionally require an adapted plan under a state intervention; the
stale strategy fails.

The new semantic recovery operator fixes a more important gap in the earlier
recursion prototype. The consequential action first returns a recoverable
semantic conflict, after which the trajectory diagnoses the observed failure,
repairs state for the exact original target, and retries the same action with
fresh repair evidence. This is counted as recovery only when a later branch
explicitly resolves the observed error code.

Recursive instructions no longer enumerate `inspect -> reserve -> commit` or
`poll -> retry` workflows. They state outcome constraints; the environment
exposes the needed mechanism. Transition and decision metrics are separate:
changing from `pending` to `ready` can count as a state-dependent transition,
but decision entropy is credited only when a valid counterfactual intervention
forces an early-grounded strategy divergence.

The corrected adaptive demonstration corpus is under
`outputs/task_first/final_wikihow_corpus_adaptive_v3/`:

| Measure | Value |
| --- | ---: |
| Unique WikiHow source tasks | 2 |
| Unique semantic episodes | 4 |
| Recursive descendants | 2 |
| Rendered training rows | 12 |
| Canonical / renamed APIs | 4 / 8 |
| Steps | 12--26 |
| Handle-chain depth | 5--9 |
| Maximum delayed handle use | 8--15 |
| Semantic episodes with semantic recovery | 2 / 4 |
| Semantic episodes with a grounded planning decision | 4 / 4 |
| Decision entropy | 1--2 bits |
| Semantic-alias rows passing identifiability gate | 8 / 8 |

This corpus is a validation sample, not a scale claim. Rendered rows are API
surface variants of four episodes, not twelve independent tasks. Its main value
is that source diversity, semantic episode diversity, recursive descendants,
and interface diversity are measured separately.

The earlier `adaptive_v2` opaque renderings are superseded. They renamed tools
to `operation_*` while some source bindings had empty descriptions, so several
same-schema tools were impossible to identify before execution even though the
private oracle could replay them. `adaptive_v3` keeps names opaque but provides
public semantic affordances in every rendered tool and parameter description;
the admission gate rejects empty or colliding public signatures.

### Source identity failure found during validation

Older extracts used the ordinal among selected rows as `id`. Consequently,
`wikihow_computer_000068` referred to different articles in the 100-row and
10k-row files. This allowed a resume run to overwrite source grounding while
reusing an unrelated environment. The task execution was valid, but the source
attribution was not.

The pipeline now binds a seed to `source_id + SHA-256(source_text)`, refuses a
mismatch during resume, and allows replacement only with explicit
`--regenerate-seed`. New extracts derive IDs from the original dataset
`source_index`; sequential IDs remain an opt-in legacy mode. The affected scan
seed and recursive lineage were regenerated from the correct source before the
final corpus was exported.

## Feasibility Boundary

The experiment validates the architecture, not production-scale yield. One
accepted lineage is enough to refute impossibility, but not enough to estimate
5k-scale cost or diversity. The current generic operators cover target-specific
audit evidence, observation-dependent execution routes, readiness lifecycles,
and one class of recoverable target-state conflict. This remains much narrower
than real desktop or service failure distributions.

Before scaling, the next required work is:

1. Estimate acceptance rate and repair cost on a stratified sample larger than
   these two accepted roots.
2. Add deterministic recursive operators for rollback/idempotency, resource
   contention, stale-object replacement, and cross-object consistency.
3. Require source/operator family quotas before model calls, then lineage,
   semantic-fingerprint, and complexity-cell quotas at admission.
4. Evaluate held-out policy models on roots and descendants; measure pass-rate
   decay and whether adapted counterfactual plans are found without oracle data.
5. Add split/merge and schema-shape API renderers. Name randomization proves
   name invariance, but not full tool-interface compositional transfer.
