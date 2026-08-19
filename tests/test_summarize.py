import json
import os
import tempfile
import unittest

from scripts.summarize import (load_rows, select_hyperparams, selected_test_rows,
                               test_summary)


def write_result(root, symbol, filename, **overrides):
    """按 control/runs/<symbol>/<method>[_w][_lam][_seed].json 布局写一个合成结果文件。"""
    payload = {"symbol": symbol, "method": "GRID", "seed": 0,
               "hindsight_weight": 0.1, "inventory_lambda": 3.0,
               "splits": {}, "TR": 0.0, "SR": 0.0, "CR": 0.0, "SoR": 0.0,
               "train_log": {"best_val_SR": 1.0}}
    payload.update(overrides)
    if payload["train_log"] is None:
        del payload["train_log"]   # 规则基线的结果文件没有训练日志字段
    folder = os.path.join(root, symbol)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        json.dump(payload, f)


class LoadRowsTest(unittest.TestCase):
    def test_reads_the_symbol_directory_layout(self):
        with tempfile.TemporaryDirectory() as root:
            write_result(root, "A", "GRID_w0.1_lam3_seed0.json")
            write_result(root, "B", "GRID_w0.1_lam3_seed1.json", seed=1,
                         train_log={"best_val_SR": 0.5})
            write_result(root, "A", "OPEN.json", method="OPEN", seed=None,
                         hindsight_weight=None, inventory_lambda=None,
                         train_log=None)

            df = load_rows(root)

            self.assertEqual(len(df), 3)
            self.assertEqual(sorted(df["symbol"]), ["A", "A", "B"])
            grid = df[df["method"] == "GRID"].set_index("seed")
            self.assertEqual(grid.loc[0, "val_SR"], 1.0)
            self.assertEqual(grid.loc[1, "val_SR"], 0.5)
            # 规则基线没有训练日志，val_SR 记为缺失
            self.assertTrue(df[df["method"] == "OPEN"]["val_SR"].isna().all())


class HyperparameterSelectionTest(unittest.TestCase):
    """跨标的与种子按 val_SR 均值选优；基线行保留；空选优表不崩。"""

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        # w=0.1 的四次运行（2 标的 × 2 种子）均值 0.85，优于 w=0.2 的 0.35
        for symbol, scores in (("A", (1.0, 0.9)), ("B", (0.8, 0.7))):
            for seed, score in enumerate(scores):
                write_result(self.root.name, symbol,
                             f"GRID_w0.1_lam3_seed{seed}.json", seed=seed,
                             train_log={"best_val_SR": score})
        for symbol, scores in (("A", (0.2, 0.3)), ("B", (0.4, 0.5))):
            for seed, score in enumerate(scores):
                write_result(self.root.name, symbol,
                             f"GRID_w0.2_lam3_seed{seed}.json", seed=seed,
                             hindsight_weight=0.2,
                             train_log={"best_val_SR": score})
        write_result(self.root.name, "A", "OPEN.json", method="OPEN", seed=None,
                     hindsight_weight=None, inventory_lambda=None, train_log=None)
        self.df = load_rows(self.root.name)

    def tearDown(self):
        self.root.cleanup()

    def test_selection_pools_symbols_and_seeds(self):
        selection = select_hyperparams(self.df)

        best = selection[selection["method"] == "GRID"].iloc[0]
        self.assertEqual(best["w"], 0.1)
        self.assertAlmostEqual(best["val_SR"], 0.85)
        self.assertEqual(best["n_runs"], 4)

    def test_selected_rows_keep_baselines_and_the_best_grid(self):
        selected = selected_test_rows(self.df, select_hyperparams(self.df))

        self.assertEqual(len(selected[selected["method"] == "OPEN"]), 1)
        grid = selected[selected["method"] == "GRID"]
        self.assertEqual(set(grid["w"]), {0.1})
        self.assertEqual(len(grid), 4)
        # 汇总表对多种子给出均值 ± 标准差
        summary = test_summary(selected)
        self.assertIn("±", summary[summary["method"] == "GRID"]["TR"].iloc[0])

    def test_empty_selection_keeps_only_rule_baselines(self):
        rules_only = self.df[self.df["val_SR"].isna()]
        selection = select_hyperparams(rules_only)

        self.assertTrue(selection.empty)
        selected = selected_test_rows(self.df, selection)
        self.assertEqual(set(selected["method"]), {"OPEN"})
        test_summary(selected)   # 空选优表下汇总不崩


if __name__ == "__main__":
    unittest.main()
