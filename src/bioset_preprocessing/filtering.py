from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .stages.overlaps import OverlapTileResult, ChannelTileStats, PairRow, SetRow


@dataclass
class FilterConfig:
    min_overlap_coeff: float = 0.0
    min_inter_vox: int = 0
    filter_by_max_dilation: bool = True
    min_tiles_present: int = 1
    top_k_percent: Optional[float] = None
    min_set_size: int = 2
    max_set_size: int = 10
    n_workers: int = 8


@dataclass
class FilterStats:
    tiles_processed: int = 0
    pairs_before: int = 0
    pairs_after: int = 0
    sets_before: int = 0
    sets_after: int = 0

    def __str__(self) -> str:
        p_pct = 100 * (1 - self.pairs_after / max(1, self.pairs_before))
        s_pct = 100 * (1 - self.sets_after / max(1, self.sets_before))
        return (
            f"FilterStats:\n"
            f"  Tiles: {self.tiles_processed}\n"
            f"  Pairs: {self.pairs_before:,} -> {self.pairs_after:,} ({p_pct:.1f}% removed)\n"
            f"  Sets: {self.sets_before:,} -> {self.sets_after:,} ({s_pct:.1f}% removed)"
        )


def _load_checkpoint_raw(filepath: Path) -> dict:
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        return json.load(f)


def _save_checkpoint_raw(filepath: Path, data: dict) -> None:
    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
        json.dump(data, f)


def _get_max_dilation(radii: List[float]) -> float:
    return max(radii)


def _collect_stats_from_file(filepath: Path, max_r: float, cfg: FilterConfig) -> dict:
    data = _load_checkpoint_raw(filepath)
    tile_x, tile_y = data['tile_x'], data['tile_y']
    pair_stats = {}
    set_stats = {}

    for p in data['pairs']:
        if p['r_um'] != max_r:
            continue
        key = (p['a'], p['b'])
        oc = p['overlap_coeff']
        inter = p['inter_vox']
        if oc < cfg.min_overlap_coeff or inter < cfg.min_inter_vox:
            continue
        if key not in pair_stats:
            pair_stats[key] = {'count': 0, 'best_oc': 0.0, 'best_tile': None}
        pair_stats[key]['count'] += 1
        if oc > pair_stats[key]['best_oc']:
            pair_stats[key]['best_oc'] = oc
            pair_stats[key]['best_tile'] = (oc, tile_x, tile_y)

    for s in data['sets']:
        if s['r_um'] != max_r:
            continue
        if s['k'] < cfg.min_set_size or s['k'] > cfg.max_set_size:
            continue
        key = tuple(s['members'])
        oc = s['overlap_coeff']
        inter = s['inter_vox']
        if oc < cfg.min_overlap_coeff or inter < cfg.min_inter_vox:
            continue
        if key not in set_stats:
            set_stats[key] = {'count': 0, 'best_oc': 0.0, 'best_tile': None}
        set_stats[key]['count'] += 1
        if oc > set_stats[key]['best_oc']:
            set_stats[key]['best_oc'] = oc
            set_stats[key]['best_tile'] = (oc, tile_x, tile_y)

    return {'filepath': str(filepath), 'pair_stats': pair_stats, 'set_stats': set_stats}


def _merge_stats(all_stats):
    pair_global = {}
    set_global = {}
    for stats in all_stats:
        for key, s in stats['pair_stats'].items():
            if key not in pair_global:
                pair_global[key] = {'count': 0, 'tiles': []}
            pair_global[key]['count'] += s['count']
            if s['best_tile']:
                pair_global[key]['tiles'].append(s['best_tile'])
        for key, s in stats['set_stats'].items():
            if key not in set_global:
                set_global[key] = {'count': 0, 'tiles': []}
            set_global[key]['count'] += s['count']
            if s['best_tile']:
                set_global[key]['tiles'].append(s['best_tile'])
    return pair_global, set_global


def _compute_top_k_tiles(global_stats, top_k_percent):
    keep_tiles = {}
    for key, stats in global_stats.items():
        tiles = stats['tiles']
        if not tiles:
            continue
        tiles_sorted = sorted(tiles, key=lambda x: x[0], reverse=True)
        k = max(1, int(len(tiles_sorted) * top_k_percent))
        keep_tiles[key] = {(t[1], t[2]) for t in tiles_sorted[:k]}
    return keep_tiles


def _filter_single_file(args):
    (filepath, max_r, cfg, valid_pairs, valid_sets, pair_keep_tiles, set_keep_tiles) = args
    data = _load_checkpoint_raw(filepath)
    tile_coord = (data['tile_x'], data['tile_y'])
    pairs_before = len(data['pairs'])
    sets_before = len(data['sets'])

    filtered_pairs = []
    for p in data['pairs']:
        key = (p['a'], p['b'])
        if key not in valid_pairs:
            continue
        if cfg.filter_by_max_dilation and p['r_um'] == max_r:
            if p['overlap_coeff'] < cfg.min_overlap_coeff or p['inter_vox'] < cfg.min_inter_vox:
                continue
            if pair_keep_tiles and key in pair_keep_tiles and tile_coord not in pair_keep_tiles[key]:
                continue
        filtered_pairs.append(p)

    filtered_sets = []
    for s in data['sets']:
        key = tuple(s['members'])
        if s['k'] < cfg.min_set_size or s['k'] > cfg.max_set_size:
            continue
        if key not in valid_sets:
            continue
        if cfg.filter_by_max_dilation and s['r_um'] == max_r:
            if s['overlap_coeff'] < cfg.min_overlap_coeff or s['inter_vox'] < cfg.min_inter_vox:
                continue
            if set_keep_tiles and key in set_keep_tiles and tile_coord not in set_keep_tiles[key]:
                continue
        filtered_sets.append(s)

    data['pairs'] = filtered_pairs
    data['sets'] = filtered_sets
    return {
        'filepath': filepath, 'data': data,
        'pairs_before': pairs_before, 'pairs_after': len(filtered_pairs),
        'sets_before': sets_before, 'sets_after': len(filtered_sets),
    }


class StreamingFilter:
    def __init__(self, config: FilterConfig):
        self.config = config
        self.stats = FilterStats()

    def filter_checkpoints(self, input_dir: Path, output_dir: Path) -> FilterStats:
        cfg = self.config
        filepaths = sorted(input_dir.glob("tile_*.json.gz"))
        n_files = len(filepaths)
        if n_files == 0:
            raise RuntimeError(f"No checkpoints found in {input_dir}")

        first_data = _load_checkpoint_raw(filepaths[0])
        max_r = _get_max_dilation(first_data['radii_um'])

        all_stats = []
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {executor.submit(_collect_stats_from_file, fp, max_r, cfg): fp for fp in filepaths}
            for future in as_completed(futures):
                all_stats.append(future.result())

        pair_global, set_global = _merge_stats(all_stats)
        valid_pairs = {k for k, v in pair_global.items() if v['count'] >= cfg.min_tiles_present}
        valid_sets = {k for k, v in set_global.items() if v['count'] >= cfg.min_tiles_present}

        pair_keep_tiles = None
        set_keep_tiles = None
        if cfg.top_k_percent is not None:
            pair_keep_tiles = _compute_top_k_tiles({k: v for k, v in pair_global.items() if k in valid_pairs}, cfg.top_k_percent)
            set_keep_tiles = _compute_top_k_tiles({k: v for k, v in set_global.items() if k in valid_sets}, cfg.top_k_percent)

        del all_stats, pair_global, set_global
        output_dir.mkdir(parents=True, exist_ok=True)

        args_list = [(fp, max_r, cfg, valid_pairs, valid_sets, pair_keep_tiles, set_keep_tiles) for fp in filepaths]
        self.stats = FilterStats(tiles_processed=n_files)

        with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {executor.submit(_filter_single_file, args): args[0] for args in args_list}
            for future in as_completed(futures):
                result = future.result()
                self.stats.pairs_before += result['pairs_before']
                self.stats.pairs_after += result['pairs_after']
                self.stats.sets_before += result['sets_before']
                self.stats.sets_after += result['sets_after']
                out_path = output_dir / Path(result['filepath']).name
                _save_checkpoint_raw(out_path, result['data'])

        return self.stats
