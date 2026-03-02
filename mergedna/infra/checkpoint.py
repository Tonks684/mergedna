from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int | None = None,
    cfg: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Save a portable checkpoint (CPU tensors) for reproducible resumption.

    Contents:
      - model_state
      - optimizer_state (optional)
      - meta: step, cfg (json), extra (json)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ckpt: dict[str, Any] = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": {
            "step": int(step) if step is not None else None,
            "cfg": _to_jsonable(cfg),
            "extra": _to_jsonable(extra or {}),
        },
    }
    if optimizer is not None:
        ckpt["optimizer_state"] = optimizer.state_dict()

    torch.save(ckpt, p)
    return p


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load checkpoint into model (+ optimizer if provided).

    Returns meta dict.
    """
    p = Path(path)
    ckpt = torch.load(p, map_location=map_location)

    model.load_state_dict(ckpt["model_state"], strict=strict)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt.get("meta", {})


def save_run_metadata(path: str | Path, meta: dict[str, Any]) -> Path:
    """
    Optional helper for saving human-readable metadata next to a checkpoint.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return p