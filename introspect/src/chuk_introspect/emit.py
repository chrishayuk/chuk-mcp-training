"""Append-only JSONL metric emission to $CHUK_METRICS.

Line shape is the worker's MetricTail contract (chuk-compute-worker
metrics.rs::parse_metric_line): one JSON object per line, a numeric ``step``
field, every other numeric field a metric value; non-numeric fields dropped.
Whole-line appends with flush keep us safe beside the trainer's own writer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import METRIC_STEP_KEY


class MetricsEmitter:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def emit(self, step: int, values: dict[str, float]) -> None:
        if not values:
            return
        record: dict[str, float | int] = {METRIC_STEP_KEY: step, **values}
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
