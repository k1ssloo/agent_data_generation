# Task Factory Performance Optimization

## Result

The optimized path keeps the complete static, execution, provenance, alternate
API, ablation, counterfactual, and recursive-preparation gates. It reduces model
work by separating semantic generation from deterministic compilation and by
repairing the approved parent instead of regenerating a complete task.

Measured on `wikihow_computer_000068`:

| Experiment | Wall time | Physical model requests | Accepted roots |
| --- | ---: | ---: | ---: |
| Early full-repair loop | 1216.426 s | 12 | 0 |
| Guarded local patch for final evidence | 10.141 s | 1 | 1 |
| Deterministic final-evidence repair | 0.116 s | 0 | 1 |
| Deterministic ordinal-provenance repair | 0.163 s | 0 | 1 |
| Deterministic counterfactual admission | 0.089 s | 0 | 1 |
| Idempotent resume of one root plus five descendants | 0.880 s | 0 | 6 |
| Weak-candidate repair before quality-aware early stop | 199.945 s | 4 | 0 |
| Same weak candidate with quality-aware early stop | 0.050 s | 0 | 0 |

The accepted quality baseline has 17 steps, delayed-handle distance 10,
handle-chain depth 5, goal-evidence coverage 1.0 from an independent read-only
observation after the final goal mutation, no unexplained arguments, two valid
counterfactuals, and necessary-action ratio 0.8235. Three sequential recursive
operators were also accepted with the parent plan invalid on every child and all
counterfactual gates passing.

For replacement generation, `--quality-baseline-dir` turns these causal metrics
into component-wise floors. A faster 11-step scanner candidate was correctly
rejected against the 17-step baseline because steps, handle depth, within-rollout
branch count, and decision entropy regressed; its higher action-ablation ratio
could not compensate for those losses.

These measurements isolate different stages. The deterministic and recursive
numbers are resume/parent-amortized throughput, not cold-start semantic planning
latency.

### WikiHow workflow amortization

A second measurement reuses model-generated Stage 3 workflows that were already
derived from WikiHow process text, then compiles the hidden environment and
applies the recursive child as deterministic code:

| Stage | Wall time | Result |
| --- | ---: | ---: |
| Compile and recursively probe 200 existing workflows | 2.042 s | 25 accepted children |
| Run the complete bundle validator on all children | 0.174 s | 25/25 valid |
| Select and export 20 distinct semantic episodes | 0.063 s | 20 rows |
| Total measured local path | 2.279 s | 20 exported rows |

This is approximately 8.8 exported semantic episodes per second, or 31,600 per
hour, for the amortized compilation path. It is not a cold raw-text-to-episode
rate: the expensive WikiHow-to-workflow semantic synthesis was paid by the
earlier Stage 3 run. Cold complete-bundle requests still showed 94--113 seconds
for failed candidates, and routed large-JSON requests remained outstanding after
roughly three minutes. Since that cold experiment accepted no episode, it has no
honest accepted-throughput speedup denominator.

The exported sample is structurally valid but should not be described as a
strong adaptive corpus: all 20 rows have complete goal evidence and zero missing
argument provenance, while only two contain an observation-dependent branch and
none contains semantic failure recovery. The fast path solves environment
compilation and validation throughput; improving decision density remains a
separate data-quality task.

A later semantic-quality audit narrowed that claim further. All 20 satisfy their
declared executable contracts, but no row is irreducible under domain-action
ablation and none has a validated planning alternative or recovery. Of 193 raw
steps, single-action ablation retains 143 while contiguous dependency-chain
minimization retains 136. Five removable domain mutations occur across four
rows. Therefore the original `goal_evidence_coverage=1.0` is now exposed as
`contract_goal_evidence_coverage=1.0`; instruction-goal coverage is explicitly
unevaluated until a semantic alignment audit passes.

A `high`-reasoning, two-candidate parallel experiment reduced bundle generation
wall time to roughly 80 seconds, but both candidates regressed against the
accepted baseline and were rejected. A later `fast` service-tier sample also
failed the corrected mutation-after-observation evidence gate. Neither result is
counted as an accepted speedup. An independent `xhigh` quality resample took
79.5 seconds and returned an even weaker five-step design. The framework now
skips generic repair for such quality regressions and rejects immediately by
default. Independent resampling remains available only through an explicit
positive `--quality-resample-candidates` value.

## Implemented Path

1. Exact content-addressed cache for parsed model JSON; rejected bundle samples
   are evicted while validated seeds and contracts remain reusable.
2. Immutable normalized contracts; candidate errors cannot be hidden by
   rewriting the task specification.
3. Restricted JSON Patch repair with full-gate acceptance and complete-repair
   fallback. Patch reasoning defaults to `medium`.
4. A deterministic fast path for the sole-error case where successful final
   domain observation omits goal evidence. It is accepted only after the full
   gate passes.
5. Immediate ordinal control arguments can be exposed by the preceding
   successful domain observation without a model call. The repair changes only
   that selected response branch and is retained only when the complete gate
   passes.
6. Invalid optional counterfactual witnesses are deterministically removed only
   when at least one strict adapted-success/stale-failure witness remains and
   the complete root gate passes after admission.
7. Concurrent roots and concurrent bundle candidates, with deterministic
   strongest-candidate selection.
8. Static-first validation, conservative state-shape completion, compact repair
   feedback, and sliced contract-repair context.
9. Deterministic recursive parent patches, counterfactual replay, API rendering,
   metrics, and export. No model call is used for these stages.
10. Stage profiles report physical requests, physical prompt/response characters,
   cache hits, latency, and accepted semantic episodes per hour.
11. Validated seed evidence remains in `seed.json`; later model stages receive a
    semantic projection without duplicated verbatim spans. JSON backends are
    instructed to emit minified output.
12. The Codex subprocess explicitly receives the configured `service_tier`, so
    a project configured for the fast tier does not silently fall back after
    moving execution into an isolated temporary directory.
13. Goal-evidence coverage is computed from one read-only domain observation
    after the last goal-state mutation, not from a union of reads across the
    whole trace. A faster candidate that made the send action read every goal
    path was rejected as mutation-as-verifier data.
14. Recursive resume is idempotent: an existing child is reused only when every
    semantic bundle field is identical; divergent content is never overwritten.
15. Candidates below a component-wise baseline are not generically patched by
    default. They are rejected immediately; a fresh independent quality
    resample is an explicit experiment. This avoids spending full-repair or
    resampling tokens on a structurally weaker parent.
16. Existing replayable WikiHow workflows can be lifted locally into a hidden
    causal runtime. Public enums are compiled only for genuine option fields;
    identifiers, names, paths, and queries are never copied into schemas as
    oracle hints. A first-step workflow context handle is consumed only by the
    final read-only outcome observation, creating delayed attribution without
    refreshing the handle at the end.

## Operating Guidance

Start with the configured Codex provider and compare accepted semantic episodes
per hour, not raw candidate rate. Increase `--workers` within endpoint limits.
Use `--bundle-candidates 2` only when the first-pass acceptance gain outweighs
the extra request. Keep the default guarded patch path and exact cache enabled.

The directly tested Responses endpoint did not improve this workload: large
structured generations truncated or timed out. It remains opt-in and should be
adopted only after a representative shard improves accepted throughput without
causal-metric regression.
