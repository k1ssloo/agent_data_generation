# rLLM Training Integration

This repository uses [rLLM](https://github.com/rllm-org/rllm) as the Agent SFT/RL
orchestrator and its veRL backend for distributed GPU training. rLLM was chosen
because one agent harness can be used for SFT, evaluation, GRPO, REINFORCE, and
on-policy distillation while retaining custom environment and evaluator code.

## Data Boundary

- `sft.jsonl` contains public OpenAI messages, public tool schemas, and aggregate
  validation metadata. It never contains initial state, contracts, reference
  plans, or verifier predicates.
- `rl_tasks.jsonl` contains the instruction and a bundle locator. At rollout
  time, `GemTaskEnvironment` resets the hidden causal runtime and exposes only
  `EpisodeRunner.policy_context()`.
- RL reward checks final goals, invariants, argument provenance, and efficiency.
  It does not require copying the private reference action sequence.

The current rLLM veRL SFT loader consumes the `messages` column but does not pass
each row's dynamic `tools` field to the Hugging Face chat template. Therefore
`export_rllm_dataset.py` also places the exact public tool catalog in the system
message. The original structured `tools` field is retained. Use
`--no-inject-tools` only with a loader that explicitly forwards dynamic tools.

## Export Existing Bundles

First validate a bundle archive:

```bash
python3 scripts/validate_task_bundles.py \
  --input-dir outputs/task_first/my_run/roots \
  --output outputs/task_first/my_run/validation.jsonl \
  --require-goal-alignment \
  --require-public-executability \
  --require-adaptive
```

Then export both training views:

```bash
python3 scripts/export_rllm_dataset.py \
  --validation outputs/task_first/my_run/validation.jsonl \
  --output-dir outputs/task_first/my_run/training
```

## Batch Generation

The batch driver shards input processes, resumes completed model stages, validates
every accepted root, and exports only passing bundles:

```bash
python3 scripts/run_task_factory_batch.py \
  --input data/wikihow_computer_10000.jsonl \
  --output-dir outputs/task_first/batch_adaptive_v1 \
  --config ../config.toml \
  --provider responses \
  --limit 1000 \
  --shard-size 25 \
  --workers 4 \
  --bundle-candidates 2 \
  --recursive-generations 2 \
  --resume \
  --continue-on-error
```

Add `--strict-vnext` for the smaller 15-25-call, 3-5-decision, recovery-rich
subset. Keep the default adaptive tier for production throughput and report the
two tiers separately.

## Install And Train

rLLM currently requires Python 3.11 or newer. Install it in a dedicated GPU
environment rather than adding its distributed dependencies to this synthesis
workspace:

```bash
uv pip install "rllm[verl] @ git+https://github.com/rllm-org/rllm.git@9beb6e0f676a46d38858991fd79ac5f8e0b16d4c"
```

SFT uses the exported `messages` records. A typical rLLM invocation is:

```bash
rllm sft \
  --train-file outputs/task_first/batch_adaptive_v1/training/sft.jsonl \
  --model Qwen/Qwen3.5-4B \
  --backend verl \
  --gpus 8 \
  --max-length 32768 \
  --tokenize-method hf_template
```

Online GRPO uses the supplied trainer entry point:

```bash
python -m training.train_rllm \
  +gem.train_file=outputs/task_first/batch_adaptive_v1/training/rl_tasks.jsonl \
  rllm/backend=verl \
  algorithm.adv_estimator=grpo \
  rllm.algorithm.use_rllm=true \
  data.train_batch_size=16 \
  +model.name=Qwen/Qwen3.5-4B \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  trainer.n_gpus_per_node=8
```

The rollout uses the training gateway at `AgentConfig.base_url`, so model token
IDs and log probabilities are captured by rLLM/veRL without exposing private
environment state to the policy. The pinned revision above is the interface
against which this adapter was tested; update the pin only after rerunning the
adapter contract tests.
