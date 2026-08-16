"""Launch online GEM agent training with rLLM.

Example::

    python -m training.train_rllm \
      +gem.train_file=outputs/my_run/training/rl_tasks.jsonl \
      rllm/backend=verl \
      +model.name=Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from rllm.data import Dataset
from rllm.trainer import AgentTrainer

from .rllm_adapter import gem_causal_evaluator, gem_tool_rollout, require_rllm
from .package import anchor_portable_rows


def load_rl_dataset(path: str) -> Dataset:
    """Load RL rows and anchor portable bundle paths to the dataset directory."""

    dataset = Dataset.load_data(path)
    anchor_portable_rows(dataset.data, path)
    return dataset


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="unified",
    version_base=None,
)
def main(config: DictConfig) -> None:
    require_rllm()
    train_file = OmegaConf.select(config, "gem.train_file")
    val_file = OmegaConf.select(config, "gem.val_file")
    if not isinstance(train_file, str) or not train_file:
        raise ValueError("set +gem.train_file=/path/to/rl_tasks.jsonl")
    train_dataset = load_rl_dataset(train_file)
    val_dataset = load_rl_dataset(val_file) if val_file else None
    if not train_dataset.data:
        raise ValueError("training dataset is empty")
    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=gem_tool_rollout,
        evaluator=gem_causal_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
