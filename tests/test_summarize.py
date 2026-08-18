import json
import os
import tempfile
import unittest

import pandas as pd

from scripts.summarize import load_rows, select_hyperparams, selected_test_rows


class LoadRowsTest(unittest.TestCase):
    def test_reads_all_fold_directories(self):
        with tempfile.TemporaryDirectory() as d:
            for fold in (0, 1):
                path = os.path.join(d, "A", f"fold_{fold}")
                os.makedirs(path)
                with open(os.path.join(path, "OPEN.json"), "w", encoding="utf-8") as f:
                    json.dump({"symbol": "A", "fold": fold, "method": "OPEN", "seed": None,
                               "hindsight_weight": None, "inventory_lambda": None,
                               "TR": 0.0, "SR": 0.0, "CR": 0.0, "SoR": 0.0, "splits": {}}, f)

            df = load_rows(d)

            self.assertEqual(sorted(df["fold"]), [0, 1])


class HyperparameterSelectionTest(unittest.TestCase):
    # 折 0 上 w=0.1 的验证均值更高，折 1 上 w=0.2 更高
    rows = [
        {"symbol": "A", "fold": 0, "method": "OPEN", "w": None, "lam": None, "val_SR": None},
        {"symbol": "A", "fold": 0, "method": "GRID", "w": 0.1, "lam": 3.0, "val_SR": 1.0},
        {"symbol": "B", "fold": 0, "method": "GRID", "w": 0.1, "lam": 3.0, "val_SR": 0.8},
        {"symbol": "A", "fold": 0, "method": "GRID", "w": 0.2, "lam": 3.0, "val_SR": 0.2},
        {"symbol": "B", "fold": 0, "method": "GRID", "w": 0.2, "lam": 3.0, "val_SR": 0.4},
        {"symbol": "A", "fold": 1, "method": "GRID", "w": 0.1, "lam": 3.0, "val_SR": 0.0},
        {"symbol": "A", "fold": 1, "method": "GRID", "w": 0.2, "lam": 3.0, "val_SR": 0.5},
    ]

    def setUp(self):
        self.frame = pd.DataFrame(self.rows)
        self.selected = selected_test_rows(self.frame, select_hyperparams(self.frame))

    def test_rule_baselines_are_always_kept(self):
        self.assertEqual(len(self.selected[self.selected["method"] == "OPEN"]), 1)

    def test_selection_pools_symbols_within_a_fold(self):
        fold0 = self.selected[(self.selected["fold"] == 0) & (self.selected["method"] == "GRID")]

        self.assertEqual(set(fold0["w"]), {0.1})
        self.assertEqual(len(fold0), 2)

    def test_folds_select_independently(self):
        fold1 = self.selected[(self.selected["fold"] == 1) & (self.selected["method"] == "GRID")]

        self.assertEqual(set(fold1["w"]), {0.2})


if __name__ == "__main__":
    unittest.main()
