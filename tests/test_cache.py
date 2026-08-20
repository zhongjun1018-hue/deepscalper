import json
import os
import tempfile
import unittest
from dataclasses import asdict

import numpy as np

from data_provider.windows import (CACHE_ARRAYS, CACHE_SCHEMA, FEATURE_NAMES,
                                   TARGET_NAMES, WindowSpec, cache_path, load_cache,
                                   predictions_current, write_predictions)

DAYS, ROWS = 2, 11   # ROWS 为合成的分钟行数（真实缓存为 MINUTES_PER_DAY）


def make_cache(cache_dir, data_dir, symbol="TEST", schema=CACHE_SCHEMA):
    """合成统一缓存：窗口块已建、preds 仍为 NaN（forecast.train 回写前的形态）。

    data_dir 为空目录，源文件签名因而是空列表，与 load_cache 的窗口级 metadata 一致。
    """
    metadata = {"schema": schema, "symbol": symbol, "source": [],
                "window": asdict(WindowSpec()), "feature_names": FEATURE_NAMES,
                "target_names": TARGET_NAMES, "prediction": None}
    targets = np.full((DAYS, ROWS, len(TARGET_NAMES)), 2.0, dtype=np.float32)
    np.savez(cache_path(symbol, cache_dir),
             _metadata=json.dumps(metadata, sort_keys=True),
             dates=np.array(["20260101", "20260102"]),
             features=np.full((DAYS, ROWS, len(FEATURE_NAMES)), 1.0, dtype=np.float32),
             targets=targets, preds=np.full_like(targets, np.nan),
             width=np.full(DAYS, 0.01, dtype=np.float32),
             anchor_ticks=np.tile(np.arange(ROWS, dtype=np.int64), (DAYS, 1)))
    os.makedirs(data_dir, exist_ok=True)


class UnifiedCacheTest(unittest.TestCase):
    def test_load_returns_every_array(self):
        with tempfile.TemporaryDirectory() as cache_dir, \
                tempfile.TemporaryDirectory() as data_dir:
            make_cache(cache_dir, data_dir)

            entry = load_cache("TEST", data_dir=data_dir, cache_dir=cache_dir,
                               zero_nan=False)

            self.assertEqual(sorted(entry), sorted(CACHE_ARRAYS))
            self.assertTrue(np.isnan(entry["preds"]).all())   # 尚未训练
            np.testing.assert_array_equal(entry["features"], 1.0)
            np.testing.assert_array_equal(entry["targets"], 2.0)
            np.testing.assert_array_equal(entry["anchor_ticks"][0], np.arange(ROWS))

    def test_write_predictions_roundtrip(self):
        with tempfile.TemporaryDirectory() as cache_dir, \
                tempfile.TemporaryDirectory() as data_dir:
            make_cache(cache_dir, data_dir)
            preds = np.arange(DAYS * ROWS * len(TARGET_NAMES),
                              dtype=np.float32).reshape(DAYS, ROWS, -1)
            preds[0, :3] = np.nan   # 回看窗口未完整观测的样本行没有预测

            write_predictions("TEST", preds, "hash-a", cache_dir=cache_dir)

            raw = load_cache("TEST", data_dir=data_dir, cache_dir=cache_dir, zero_nan=False)
            np.testing.assert_array_equal(raw["preds"][1], preds[1])
            self.assertTrue(np.isnan(raw["preds"][0, :3]).all())
            # 窗口块与窗口级 metadata 不受回写影响
            np.testing.assert_array_equal(raw["features"], 1.0)
            np.testing.assert_allclose(raw["width"], 0.01)

            zeroed = load_cache("TEST", data_dir=data_dir, cache_dir=cache_dir)
            np.testing.assert_array_equal(zeroed["preds"][0, :3], 0.0)

    def test_predictions_current_tracks_data_hash(self):
        with tempfile.TemporaryDirectory() as cache_dir, \
                tempfile.TemporaryDirectory() as data_dir:
            self.assertFalse(predictions_current("TEST", "hash-a", cache_dir))
            make_cache(cache_dir, data_dir)
            self.assertFalse(predictions_current("TEST", "hash-a", cache_dir))

            write_predictions("TEST", np.zeros((DAYS, ROWS, len(TARGET_NAMES))),
                              "hash-a", cache_dir=cache_dir)

            self.assertTrue(predictions_current("TEST", "hash-a", cache_dir))
            self.assertFalse(predictions_current("TEST", "hash-b", cache_dir))

    def test_stale_window_metadata_forces_rebuild(self):
        """窗口级 metadata 不匹配时整体重建；空 data_dir 下重建即报缺源数据。"""
        with tempfile.TemporaryDirectory() as cache_dir, \
                tempfile.TemporaryDirectory() as data_dir:
            make_cache(cache_dir, data_dir, schema=CACHE_SCHEMA - 1)
            with self.assertRaises(FileNotFoundError):
                load_cache("TEST", data_dir=data_dir, cache_dir=cache_dir)


if __name__ == "__main__":
    unittest.main()
