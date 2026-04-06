"""
Validation & Comparison
=======================

Compares BioSET pipeline outputs between modes (legacy vs MCMICRO) or between
runs.  Used in Phase 7 of the migration plan to verify that the MCMICRO-mode
pipeline produces scientifically comparable results to the legacy baseline.

Success criteria (configurable):
- Top-50 pairs by IoU must overlap >= 80% between modes
- Global IoU values must correlate with Pearson r > 0.90
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _open_bioset(path: str | Path) -> sqlite3.Connection:
    """Decompress a .bioset file and return a SQLite connection."""
    path = Path(path)
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    with gzip.open(path, "rb") as f_in:
        with open(tmp.name, "wb") as f_out:
            f_out.write(f_in.read())

    conn = sqlite3.connect(tmp.name)
    conn.row_factory = sqlite3.Row
    return conn


def _get_top_pairs(
    conn: sqlite3.Connection,
    dilation: float,
    hierarchy_level: int = 0,
    top_k: int = 50,
) -> List[Dict]:
    """Get top-K pairs by global IoU."""
    cursor = conn.execute("""
        SELECT
            channels,
            SUM(total_count) AS sum_inter,
            SUM(total_union) AS sum_union,
            CAST(SUM(total_count) AS REAL) / SUM(total_union) AS global_iou
        FROM combinations
        WHERE dilation = ? AND hierarchy_level = ? AND channel_count = 2
        GROUP BY channels
        HAVING sum_union > 0
        ORDER BY global_iou DESC
        LIMIT ?
    """, (dilation, hierarchy_level, top_k))

    return [dict(row) for row in cursor.fetchall()]


def _get_channel_voxels(
    conn: sqlite3.Connection,
    dilation: float = 0.0,
    hierarchy_level: int = 0,
) -> Dict[str, int]:
    """Get total voxel count per channel."""
    cursor = conn.execute("""
        SELECT channel, SUM(voxel_count) AS total
        FROM channel_stats
        WHERE dilation = ? AND hierarchy_level = ?
        GROUP BY channel
        ORDER BY total DESC
    """, (dilation, hierarchy_level))

    return {row["channel"]: row["total"] for row in cursor.fetchall()}


def _get_tile_heatmap(
    conn: sqlite3.Connection,
    channels: str,
    dilation: float,
    hierarchy_level: int = 0,
) -> Dict[Tuple[int, int], int]:
    """Get per-tile intersection counts for a channel pair."""
    cursor = conn.execute("""
        SELECT t.tile_x0, t.tile_y0, t.inter_count
        FROM combinations c
        JOIN tiles t ON c.id = t.combination_id
        WHERE c.channels = ? AND c.dilation = ? AND c.hierarchy_level = ?
    """, (channels, dilation, hierarchy_level))

    return {(row["tile_x0"], row["tile_y0"]): row["inter_count"]
            for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Results of comparing two .bioset files."""

    # Top-pair overlap
    top_pair_overlap_fraction: float = 0.0
    baseline_top_pairs: List[str] = field(default_factory=list)
    candidate_top_pairs: List[str] = field(default_factory=list)
    shared_pairs: List[str] = field(default_factory=list)

    # IoU correlation
    iou_pearson_r: float = 0.0
    n_shared_for_correlation: int = 0

    # Channel voxel comparison
    channel_voxel_correlation: float = 0.0
    channel_voxel_diffs: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Spatial heatmap correlation (per top pair)
    heatmap_correlations: Dict[str, float] = field(default_factory=dict)

    # Overall verdict
    passed: bool = False
    reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VALIDATION COMPARISON REPORT",
            "=" * 60,
            f"Top-pair overlap: {self.top_pair_overlap_fraction:.1%} "
            f"({len(self.shared_pairs)}/{len(self.baseline_top_pairs)} shared)",
            f"IoU Pearson r: {self.iou_pearson_r:.4f} "
            f"(over {self.n_shared_for_correlation} shared pairs)",
            f"Channel voxel correlation: {self.channel_voxel_correlation:.4f}",
            "",
            f"Spatial heatmap correlations (top pairs):",
        ]
        for pair, r in sorted(
            self.heatmap_correlations.items(), key=lambda x: -x[1]
        )[:10]:
            lines.append(f"  {pair}: r = {r:.4f}")

        lines.append("")
        lines.append(f"VERDICT: {'PASS' if self.passed else 'FAIL'}")
        if not self.passed:
            lines.append("Reasons for failure:")
            for r in self.reasons:
                lines.append(f"  - {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


def compare_biosets(
    baseline_path: str | Path,
    candidate_path: str | Path,
    dilation: float = 2.0,
    top_k: int = 50,
    min_overlap_fraction: float = 0.8,
    min_pearson_r: float = 0.90,
) -> ComparisonResult:
    """
    Compare two .bioset files and compute quality metrics.

    Parameters
    ----------
    baseline_path : str or Path
        Path to the baseline .bioset file (from legacy pipeline).
    candidate_path : str or Path
        Path to the candidate .bioset file (from MCMICRO pipeline).
    dilation : float
        Dilation radius to compare at.
    top_k : int
        Number of top pairs to compare.
    min_overlap_fraction : float
        Minimum fraction of top pairs that must be shared.
    min_pearson_r : float
        Minimum Pearson correlation for IoU values.

    Returns
    -------
    ComparisonResult
        Detailed comparison results with pass/fail verdict.
    """
    result = ComparisonResult()

    conn_base = _open_bioset(baseline_path)
    conn_cand = _open_bioset(candidate_path)

    try:
        # 1) Top-pair overlap
        base_pairs = _get_top_pairs(conn_base, dilation, top_k=top_k)
        cand_pairs = _get_top_pairs(conn_cand, dilation, top_k=top_k)

        base_set = {p["channels"] for p in base_pairs}
        cand_set = {p["channels"] for p in cand_pairs}
        shared = base_set & cand_set

        result.baseline_top_pairs = [p["channels"] for p in base_pairs]
        result.candidate_top_pairs = [p["channels"] for p in cand_pairs]
        result.shared_pairs = sorted(shared)
        result.top_pair_overlap_fraction = (
            len(shared) / max(1, len(base_set))
        )

        # 2) IoU correlation on shared pairs
        if len(shared) >= 3:
            base_iou = {p["channels"]: p["global_iou"] for p in base_pairs}
            cand_iou = {p["channels"]: p["global_iou"] for p in cand_pairs}

            shared_list = sorted(shared)
            b_vals = np.array([base_iou[s] for s in shared_list])
            c_vals = np.array([cand_iou[s] for s in shared_list])

            if np.std(b_vals) > 0 and np.std(c_vals) > 0:
                result.iou_pearson_r = float(np.corrcoef(b_vals, c_vals)[0, 1])
            else:
                result.iou_pearson_r = 1.0 if np.allclose(b_vals, c_vals) else 0.0

            result.n_shared_for_correlation = len(shared_list)

        # 3) Channel voxel comparison
        base_vox = _get_channel_voxels(conn_base)
        cand_vox = _get_channel_voxels(conn_cand)

        common_channels = set(base_vox.keys()) & set(cand_vox.keys())
        if len(common_channels) >= 3:
            ch_list = sorted(common_channels)
            b_v = np.array([base_vox[c] for c in ch_list], dtype=float)
            c_v = np.array([cand_vox[c] for c in ch_list], dtype=float)

            if np.std(b_v) > 0 and np.std(c_v) > 0:
                result.channel_voxel_correlation = float(
                    np.corrcoef(b_v, c_v)[0, 1]
                )
            result.channel_voxel_diffs = {
                ch: (base_vox[ch], cand_vox[ch]) for ch in ch_list
            }

        # 4) Spatial heatmap correlation for top shared pairs
        for pair_name in list(shared)[:10]:
            base_heatmap = _get_tile_heatmap(conn_base, pair_name, dilation)
            cand_heatmap = _get_tile_heatmap(conn_cand, pair_name, dilation)

            common_tiles = set(base_heatmap.keys()) & set(cand_heatmap.keys())
            if len(common_tiles) >= 5:
                tiles = sorted(common_tiles)
                b_h = np.array([base_heatmap[t] for t in tiles], dtype=float)
                c_h = np.array([cand_heatmap[t] for t in tiles], dtype=float)

                if np.std(b_h) > 0 and np.std(c_h) > 0:
                    r = float(np.corrcoef(b_h, c_h)[0, 1])
                else:
                    r = 1.0 if np.allclose(b_h, c_h) else 0.0
                result.heatmap_correlations[pair_name] = r

        # 5) Verdict
        reasons = []
        if result.top_pair_overlap_fraction < min_overlap_fraction:
            reasons.append(
                f"Top-pair overlap {result.top_pair_overlap_fraction:.1%} "
                f"< {min_overlap_fraction:.1%} threshold"
            )
        if result.iou_pearson_r < min_pearson_r and result.n_shared_for_correlation >= 3:
            reasons.append(
                f"IoU Pearson r = {result.iou_pearson_r:.4f} "
                f"< {min_pearson_r:.2f} threshold"
            )

        result.reasons = reasons
        result.passed = len(reasons) == 0

    finally:
        conn_base.close()
        conn_cand.close()

    return result


def compare_checkpoints(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    max_tiles: int = 100,
) -> Dict:
    """
    Quick comparison of checkpoint files between two runs.

    Returns summary statistics without requiring full aggregation.
    """
    from .checkpoint import load_all_checkpoints

    base_results = load_all_checkpoints(Path(baseline_dir))
    cand_results = load_all_checkpoints(Path(candidate_dir))

    base_tiles = {(r.tile_x, r.tile_y): r for r in base_results}
    cand_tiles = {(r.tile_x, r.tile_y): r for r in cand_results}

    common = set(base_tiles.keys()) & set(cand_tiles.keys())

    pair_count_diffs = []
    active_ch_diffs = []

    for key in sorted(common)[:max_tiles]:
        br = base_tiles[key]
        cr = cand_tiles[key]
        pair_count_diffs.append(abs(len(br.pairs) - len(cr.pairs)))
        active_ch_diffs.append(abs(br.n_active_channels - cr.n_active_channels))

    return {
        "baseline_tiles": len(base_tiles),
        "candidate_tiles": len(cand_tiles),
        "common_tiles": len(common),
        "tiles_compared": min(max_tiles, len(common)),
        "mean_pair_count_diff": float(np.mean(pair_count_diffs)) if pair_count_diffs else 0,
        "mean_active_ch_diff": float(np.mean(active_ch_diffs)) if active_ch_diffs else 0,
    }
