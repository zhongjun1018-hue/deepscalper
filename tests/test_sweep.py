import json
import os
import tempfile
import unittest

from control.config import Config
from control.sweep import (SWEEP_LADDERS, combo_table, job_stem, load_rows,
                           make_combo_jobs, make_jobs, parse_combo, sweep_table)


def write_result(root, param, value, seed, val_sr, tr=0.0):
    """按 control/runs/sweep/ 布局写一个合成探索结果文件。"""
    payload = {"param": param, "value": value, "seed": seed,
               "TR": tr, "SR": 0.0, "CR": 0.0, "SoR": 0.0,
               "train_log": {"best_val_SR": val_sr}}
    os.makedirs(root, exist_ok=True)
    stem = job_stem({"param": param, "value": value, "seed": seed})
    with open(os.path.join(root, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)


class MakeJobsTest(unittest.TestCase):
    """作业矩阵：全部梯子共用同一中心点作业，只展开各梯子的非默认档。"""

    def test_ladders_center_on_the_default_config(self):
        defaults = Config()
        for param, ladder in SWEEP_LADDERS.items():
            self.assertIn(getattr(defaults, param), ladder, msg=param)

    def test_center_is_shared_and_defaults_are_not_duplicated(self):
        defaults = Config()
        jobs = make_jobs((0,), list(SWEEP_LADDERS))

        self.assertEqual(sum(j["param"] is None for j in jobs), 1)
        varied = [j for j in jobs if j["param"] is not None]
        self.assertEqual(len(varied),
                         sum(len(ladder) - 1 for ladder in SWEEP_LADDERS.values()))
        for job in varied:
            self.assertNotEqual(job["value"], getattr(defaults, job["param"]))

    def test_job_stem_names(self):
        self.assertEqual(job_stem({"param": None, "value": None, "seed": 0}),
                         "default_seed0")
        self.assertEqual(job_stem({"param": "lr", "value": 3e-4, "seed": 1}),
                         "lr_0.0003_seed1")
        combo = {"lr": 3e-4, "target_sync": 4000}
        self.assertEqual(job_stem({"param": "combo", "value": combo, "seed": 0}),
                         "combo_lr0.0003+target_sync4000_seed0")


class ComboTest(unittest.TestCase):
    """组合确认：PARAM=VALUE 解析（类型随 Config 字段、按名排序）与配对对照。"""

    def test_parse_combo_types_and_order(self):
        overrides = parse_combo(["target_sync=4000", "lr=3e-4"])

        self.assertEqual(overrides, {"lr": 3e-4, "target_sync": 4000})
        self.assertIsInstance(overrides["target_sync"], int)
        self.assertEqual(list(overrides), ["lr", "target_sync"])  # 与书写顺序无关
        with self.assertRaises(ValueError):
            parse_combo(["no_such_param=1"])
        with self.assertRaises(ValueError):
            parse_combo(["lr"])

    def test_combo_jobs_pair_center_and_combo(self):
        overrides = {"target_sync": 4000}
        jobs = make_combo_jobs((0, 1), overrides)

        self.assertEqual(len(jobs), 4)   # 2 种子 × (中心点 + 组合)
        self.assertEqual(sum(j["param"] is None for j in jobs), 2)
        self.assertTrue(all(j["value"] == overrides
                            for j in jobs if j["param"] == "combo"))

    def test_combo_table_pairs_by_seed(self):
        overrides = {"target_sync": 4000}
        with tempfile.TemporaryDirectory() as root:
            for seed, (center_sr, combo_sr) in enumerate(((0.1, 0.3), (0.2, 0.1))):
                write_result(root, None, None, seed, center_sr)
                write_result(root, "combo", overrides, seed, combo_sr)
            write_result(root, None, None, 2, 0.9)   # 组合侧缺 seed2：不配对

            table, pairs, wins = combo_table(load_rows(root), overrides)

            self.assertEqual((pairs, wins), (2, 1))
            rows = table.set_index("config")
            self.assertAlmostEqual(rows.loc["默认", "val_SR"], 0.15)
            self.assertAlmostEqual(rows.loc["组合", "val_SR"], 0.2)
            self.assertAlmostEqual(rows.loc["配对差", "val_SR"], 0.05)


class SweepTableTest(unittest.TestCase):
    """汇总：中心点并入每个参数的默认档（值后缀 *），跨种子取均值。"""

    def test_center_joins_every_ladder_and_means_pool_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            for seed, (center_sr, varied_sr) in enumerate(((0.2, 0.5), (0.4, 0.7))):
                write_result(root, None, None, seed, center_sr)
                write_result(root, "lr", 1e-4, seed, varied_sr)

            table = sweep_table(load_rows(root), ["lr", "batch_size"])

            lr = table[table["param"] == "lr"].set_index("value")
            self.assertAlmostEqual(lr.loc["0.0003*", "val_SR"], 0.3)
            self.assertAlmostEqual(lr.loc["0.0001", "val_SR"], 0.6)
            self.assertEqual(lr.loc["0.0001", "n_runs"], 2)
            # 未跑过的梯子只有中心点一行（默认档）
            batch = table[table["param"] == "batch_size"]
            self.assertEqual(batch["value"].tolist(), ["128*"])


if __name__ == "__main__":
    unittest.main()
