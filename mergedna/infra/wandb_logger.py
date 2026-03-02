from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WandbConfig:
    mode: str = "disabled"  # "disabled" | "offline" | "online"
    project: str = "mergedna"
    entity: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    run_dir: str | None = None  # optional local dir


class WandbLogger:
    """
    Minimal W&B logger:
      - init() once
      - log(metrics, step=...)
      - finish()

    If mode="disabled", all methods are no-ops.
    """

    def __init__(self, cfg: WandbConfig):
        self.cfg = cfg
        self._run = None
        self._wandb = None

        if cfg.mode == "disabled":
            return

        try:
            import wandb  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "W&B is enabled but wandb is not installed. "
                "Install with `uv sync --extra wandb` (or add wandb to dependencies)."
            ) from e

        self._wandb = wandb

        # wandb.init(mode=...) supports "offline"/"online"/"disabled"
        init_kwargs: dict[str, Any] = {
            "project": cfg.project,
            "entity": cfg.entity,
            "name": cfg.name,
            "tags": cfg.tags,
            "mode": cfg.mode,
        }
        if cfg.run_dir is not None:
            init_kwargs["dir"] = cfg.run_dir

        self._run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.cfg.mode == "disabled":
            return
        assert self._wandb is not None
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.cfg.mode == "disabled":
            return
        if self._run is not None:
            self._run.finish()
            self._run = None