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

    def __init__(self, cfg: WandbConfig):

        self.mode = cfg.mode
        self.run = None
        self.enabled = False

        if self.mode == "disabled":
            return

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            if self.mode == "online":
                raise RuntimeError(
                    "W&B mode is 'online' but wandb is not installed.\n"
                    "Run: uv sync --extra wandb"
                )
            # offline → silently disable
            print("[wandb] wandb not installed — logging disabled", flush=True)
            return

        self.run = self._wandb.init(
            project=cfg.project,
            name=cfg.name,
            tags=cfg.tags,
            dir=cfg.run_dir,
            mode=self.mode,
            reinit=True,
        )
        self.enabled = True
    
    def log(self, payload: dict, step: int):
        if not self.enabled:
            return
        self.run.log(payload, step=step) 
    
    def finish(self):
        if not self.enabled:
            return
        self.run.finish()
