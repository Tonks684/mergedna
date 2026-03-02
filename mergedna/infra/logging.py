from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_safe(x: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    # torch scalars / numpy scalars
    try:
        import torch  # local import to avoid hard dependency in tooling contexts

        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().item()
            return x.detach().tolist()
    except Exception:
        pass

    # basic python numerics/strings/bools/dicts/lists
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]

    # fall back to string
    return str(x)


@dataclass
class RunLogger:
    """
    Minimal structured logger:
      - always prints JSON lines to stdout
      - optionally writes runs/<run_id>/metrics.jsonl

    This is intentionally tiny and dependency-free. W&B comes later.
    """

    run_dir: Path | None = None
    print_jsonl: bool = True
    write_jsonl: bool = True

    _t0: float = 0.0
    _last_t: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.time()
        self._last_t = self._t0

        if self.run_dir is not None and self.write_jsonl:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "metrics.jsonl").touch(exist_ok=True)

    @property
    def metrics_path(self) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / "metrics.jsonl"

    def log(self, step: int, **metrics: Any) -> dict[str, Any]:
        """
        Emit a structured metrics event. Returns the payload (useful for tests).
        """
        now = time.time()
        payload: dict[str, Any] = {
            "step": int(step),
            "time": now,
            "elapsed_s": now - self._t0,
            **{k: _json_safe(v) for k, v in metrics.items()},
        }

        line = json.dumps(payload, sort_keys=True)

        if self.print_jsonl:
            print(line, flush=True)

        if self.write_jsonl and self.metrics_path is not None:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        self._last_t = now
        return payload


def make_run_logger(
    *,
    run_root: str | os.PathLike = "runs",
    run_id: str | None = None,
    enable_file: bool = True,
    enable_print: bool = True,
) -> RunLogger:
    """
    Factory that chooses a run directory like:
      runs/<run_id>/
    If run_id is None, uses a timestamp-based id.
    """
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    run_dir = Path(run_root) / run_id if enable_file else None
    return RunLogger(run_dir=run_dir, print_jsonl=enable_print, write_jsonl=enable_file)


class ThroughputMeter:
    """
    Lightweight tokens/sec meter. You tell it how many tokens were processed
    in a step (usually B*N), and it returns instantaneous and EMA throughput.
    """

    def __init__(self, ema_beta: float = 0.9):
        self.ema_beta = float(ema_beta)
        self._t_last = time.time()
        self._ema = None

    def update(self, tokens: int) -> dict[str, float]:
        now = time.time()
        dt = max(now - self._t_last, 1e-9)
        tps = float(tokens) / dt

        if self._ema is None:
            ema = tps
        else:
            ema = self.ema_beta * self._ema + (1.0 - self.ema_beta) * tps

        self._ema = ema
        self._t_last = now
        return {"tokens_per_s": tps, "tokens_per_s_ema": ema}