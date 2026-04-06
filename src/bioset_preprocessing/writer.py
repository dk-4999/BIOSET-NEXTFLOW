from __future__ import annotations

import gzip
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

from .aggregation import HierarchyLevel, AggregatedChannelStats, AggregatedPair, AggregatedSet


class BiosetWriter:
    def __init__(self, output_path, channel_names, dilation_amounts, volume_shape):
        self.output_path = Path(output_path)
        self.channel_names = channel_names
        self.dilation_amounts = list(dilation_amounts)
        self.volume_shape = volume_shape
        self._temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._temp_db.close()
        self._conn = sqlite3.connect(self._temp_db.name)
        self._setup_schema()

    def _setup_schema(self):
        cur = self._conn.cursor()
        cur.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("""CREATE TABLE combinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, channels TEXT, channel_count INTEGER,
            dilation REAL, hierarchy_level INTEGER, total_count INTEGER, total_union INTEGER,
            iou REAL, overlap_coeff REAL)""")
        cur.execute("""CREATE TABLE tiles (
            combination_id INTEGER, tile_x0 INTEGER, tile_x1 INTEGER,
            tile_y0 INTEGER, tile_y1 INTEGER, inter_count INTEGER, union_count INTEGER,
            FOREIGN KEY (combination_id) REFERENCES combinations(id))""")
        cur.execute("""CREATE TABLE channel_stats (
            channel TEXT, channel_idx INTEGER, dilation REAL, hierarchy_level INTEGER,
            tile_x0 INTEGER, tile_x1 INTEGER, tile_y0 INTEGER, tile_y1 INTEGER,
            voxel_count INTEGER, sum_intensity REAL, mean_intensity REAL)""")
        cur.execute("CREATE INDEX idx_combinations_channels ON combinations(channels)")
        cur.execute("CREATE INDEX idx_combinations_dilation ON combinations(dilation)")
        cur.execute("CREATE INDEX idx_combinations_level ON combinations(hierarchy_level)")
        cur.execute("CREATE INDEX idx_combinations_dilation_level ON combinations(dilation, hierarchy_level)")
        cur.execute("CREATE INDEX idx_tiles_combination ON tiles(combination_id)")
        cur.execute("CREATE INDEX idx_tiles_spatial ON tiles(tile_x0, tile_y0)")
        cur.execute("CREATE INDEX idx_channel_stats_channel ON channel_stats(channel)")
        cur.execute("CREATE INDEX idx_channel_stats_level ON channel_stats(hierarchy_level)")
        self._conn.commit()

    def _channel_idx_to_name(self, idx):
        if idx < len(self.channel_names):
            return self.channel_names[idx]
        return f"ch{idx}"

    def _make_channels_key(self, indices):
        names = [self._channel_idx_to_name(i) for i in sorted(indices)]
        return "|".join(names)

    def write_metadata(self, hierarchy_levels):
        cur = self._conn.cursor()
        metadata = {
            "channels": self.channel_names,
            "hierarchy_levels": hierarchy_levels,
            "dilation_amounts": self.dilation_amounts,
            "volume_bounds": {"z": [0, self.volume_shape[0]], "y": [0, self.volume_shape[1]], "x": [0, self.volume_shape[2]]},
        }
        for key, value in metadata.items():
            cur.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self._conn.commit()

    def write_hierarchy_level(self, level):
        cur = self._conn.cursor()
        for cs in level.channels:
            mean_intensity = cs.sum_intensity / cs.voxel_count if cs.voxel_count > 0 else 0.0
            cur.execute(
                "INSERT INTO channel_stats (channel, channel_idx, dilation, hierarchy_level, tile_x0, tile_x1, tile_y0, tile_y1, voxel_count, sum_intensity, mean_intensity) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self._channel_idx_to_name(cs.channel), cs.channel, cs.r_um, cs.hierarchy_level, cs.tile_x0, cs.tile_x1, cs.tile_y0, cs.tile_y1, cs.voxel_count, cs.sum_intensity, mean_intensity))

        for pr in level.pairs:
            channels_key = self._make_channels_key([pr.a, pr.b])
            cur.execute(
                "INSERT INTO combinations (channels, channel_count, dilation, hierarchy_level, total_count, total_union, iou, overlap_coeff) VALUES (?,?,?,?,?,?,?,?)",
                (channels_key, 2, pr.r_um, pr.hierarchy_level, pr.inter_vox, pr.union_vox, pr.iou, pr.overlap_coeff))
            cid = cur.lastrowid
            cur.execute(
                "INSERT INTO tiles (combination_id, tile_x0, tile_x1, tile_y0, tile_y1, inter_count, union_count) VALUES (?,?,?,?,?,?,?)",
                (cid, pr.tile_x0, pr.tile_x1, pr.tile_y0, pr.tile_y1, pr.inter_vox, pr.union_vox))

        for sr in level.sets:
            channels_key = self._make_channels_key(sr.members)
            cur.execute(
                "INSERT INTO combinations (channels, channel_count, dilation, hierarchy_level, total_count, total_union, iou, overlap_coeff) VALUES (?,?,?,?,?,?,?,?)",
                (channels_key, sr.k, sr.r_um, sr.hierarchy_level, sr.inter_vox, sr.union_vox, sr.iou, sr.overlap_coeff))
            cid = cur.lastrowid
            cur.execute(
                "INSERT INTO tiles (combination_id, tile_x0, tile_x1, tile_y0, tile_y1, inter_count, union_count) VALUES (?,?,?,?,?,?,?)",
                (cid, sr.tile_x0, sr.tile_x1, sr.tile_y0, sr.tile_y1, sr.inter_vox, sr.union_vox))
        self._conn.commit()

    def finalize(self):
        self._conn.close()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._temp_db.name, 'rb') as f_in:
            with gzip.open(self.output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        Path(self._temp_db.name).unlink()
        return self.output_path
