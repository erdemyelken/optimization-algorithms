"""
Export utilities — CSV, JSON, PNG/SVG for results and figures.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.result import AggregatedResult, OptimizationResult


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Exporter:
    """
    Handles exporting results and figures to disk.

    Parameters
    ----------
    output_dir : str
        Root directory for all exports.  Defaults to ``results/``.
    """

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        _ensure_dir(output_dir)

    # ------------------------------------------------------------------ #
    # Results export
    # ------------------------------------------------------------------ #

    def to_csv(
        self,
        rows: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ) -> str:
        """Save a list of dicts as CSV.  Returns the file path."""
        filename = filename or f"results_{_timestamp()}.csv"
        path = os.path.join(self.output_dir, filename)
        if not rows:
            return path
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def to_json(
        self,
        data: Any,
        filename: Optional[str] = None,
    ) -> str:
        """Serialise ``data`` to JSON.  Returns the file path."""
        filename = filename or f"results_{_timestamp()}.json"
        path = os.path.join(self.output_dir, filename)

        def _default(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_default)
        return path

    def aggregated_to_csv(
        self,
        agg_results: List[AggregatedResult],
        filename: Optional[str] = None,
    ) -> str:
        from core.metrics import summary_table
        rows = summary_table(agg_results)
        return self.to_csv(rows, filename)

    def convergence_to_csv(
        self,
        agg_results: List[AggregatedResult],
        filename: Optional[str] = None,
    ) -> str:
        """Export all convergence histories to a wide CSV."""
        filename = filename or f"convergence_{_timestamp()}.csv"
        path = os.path.join(self.output_dir, filename)
        max_iter = max(len(r.convergence_mean) for r in agg_results)
        with open(path, "w", newline="", encoding="utf-8") as f:
            header = ["Iteration"] + [r.algorithm_name for r in agg_results]
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(max_iter):
                row = [i + 1]
                for r in agg_results:
                    val = r.convergence_mean[i] if i < len(r.convergence_mean) else ""
                    row.append(val)
                writer.writerow(row)
        return path

    # ------------------------------------------------------------------ #
    # Figure export
    # ------------------------------------------------------------------ #

    def save_figure(
        self,
        fig: Any,
        filename: Optional[str] = None,
        fmt: str = "png",
        dpi: int = 150,
    ) -> str:
        """
        Save a matplotlib Figure to disk.

        Parameters
        ----------
        fig :
            A ``matplotlib.figure.Figure`` instance.
        fmt :
            ``"png"`` or ``"svg"``.
        """
        filename = filename or f"figure_{_timestamp()}.{fmt}"
        if not filename.endswith(f".{fmt}"):
            filename = f"{filename}.{fmt}"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def config_snapshot(
        self,
        config: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> str:
        """Save a YAML-style config snapshot as JSON."""
        filename = filename or f"config_{_timestamp()}.json"
        return self.to_json(config, filename)
