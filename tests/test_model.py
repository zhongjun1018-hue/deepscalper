import tempfile
import unittest

import numpy as np

from data_provider.windows import FEATURE_NAMES, TARGET_NAMES
from forecast.config import Config
from forecast.model import Model

# 小规模超参：测试只验证契约与往返，不验证拟合质量
MODEL_KWARGS = {"n_estimators": 5, "early_stopping_rounds": 2}
NUM_FEATURES = len(FEATURE_NAMES) + 1   # 47 维窗口特征 + symbol_id 分类列


def synthetic_rows(n=120, seed=0):
    """(n, 48) 特征（末列为 symbol_id）与 (n, 5) 目标：目标与特征线性相关。"""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, NUM_FEATURES)).astype(np.float32)
    features[:, -1] = rng.integers(0, 3, n)
    targets = np.column_stack(
        [features[:, k] + 0.1 * k for k in range(len(TARGET_NAMES))]
    ).astype(np.float32)
    return features, targets


class ModelContractTest(unittest.TestCase):
    def test_default_config_values(self):
        self.assertEqual(Config().window.atr_mult, 0.1)
        self.assertEqual(Config().seed, 2021)

    def test_training_objective_cannot_be_overridden(self):
        for key, value in (("objective", "regression"), ("metric", "l2")):
            with self.assertRaises(ValueError, msg=key):
                Model(model_kwargs={key: value})

    def test_input_contract(self):
        model = Model(model_kwargs=MODEL_KWARGS)
        self.assertEqual(model.feature_names, FEATURE_NAMES + ["symbol_id"])
        self.assertEqual(model.label_names, TARGET_NAMES)

        features, targets = synthetic_rows()
        model.fit(features[:100], targets[:100], eval_set=(features[100:], targets[100:]))
        self.assertEqual([booster.params["objective"] for booster in model.boosters],
                         ["regression"] * len(TARGET_NAMES))

    def test_fit_predict_save_load_roundtrip(self):
        features, targets = synthetic_rows()
        model = Model(seed=7, data_hash="test", model_kwargs=MODEL_KWARGS)
        model.fit(features[:100], targets[:100], eval_set=(features[100:], targets[100:]))

        prediction = model.predict(features)
        self.assertEqual(prediction.shape, targets.shape)
        self.assertEqual(prediction.dtype, np.float32)

        with tempfile.TemporaryDirectory() as folder:
            model.save(folder)
            restored = Model(seed=7, data_hash="test", model_kwargs=MODEL_KWARGS).load(folder)
            np.testing.assert_array_equal(restored.predict(features), prediction)
            # 元数据不一致的检查点拒绝加载
            with self.assertRaises(ValueError):
                Model(seed=8, data_hash="test", model_kwargs=MODEL_KWARGS).load(folder)

    def test_seed_determinism(self):
        features, targets = synthetic_rows()
        first = Model(seed=7, model_kwargs=MODEL_KWARGS).fit(
            features[:100], targets[:100], eval_set=(features[100:], targets[100:]))
        second = Model(seed=7, model_kwargs=MODEL_KWARGS).fit(
            features[:100], targets[:100], eval_set=(features[100:], targets[100:]))

        np.testing.assert_array_equal(first.predict(features), second.predict(features))

    def test_feature_shape_is_checked(self):
        model = Model(model_kwargs=MODEL_KWARGS)
        features, targets = synthetic_rows()
        model.fit(features[:100], targets[:100], eval_set=(features[100:], targets[100:]))
        with self.assertRaises(ValueError):
            model.predict(features[:, :10])


if __name__ == "__main__":
    unittest.main()
