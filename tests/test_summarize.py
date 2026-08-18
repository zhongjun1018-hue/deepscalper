import unittest

import pandas as pd

from scripts.summarize import select_hyperparams, selected_test_rows


class HyperparameterSelectionTest(unittest.TestCase):
    def test_test_rows_keep_only_validation_selected_configuration(self):
        rows = [
            {"symbol": "A", "method": "OPEN", "w": None, "lam": None, "val_SR": None},
            {"symbol": "A", "method": "GRID", "w": 0.1, "lam": 3.0, "val_SR": 1.0},
            {"symbol": "B", "method": "GRID", "w": 0.1, "lam": 3.0, "val_SR": 0.8},
            {"symbol": "A", "method": "GRID", "w": 0.2, "lam": 3.0, "val_SR": 0.2},
            {"symbol": "B", "method": "GRID", "w": 0.2, "lam": 3.0, "val_SR": 0.4},
        ]
        frame = pd.DataFrame(rows)

        selected = selected_test_rows(frame, select_hyperparams(frame))

        self.assertEqual(len(selected), 3)
        self.assertEqual(set(selected[selected["method"] == "GRID"]["w"]), {0.1})


if __name__ == "__main__":
    unittest.main()
