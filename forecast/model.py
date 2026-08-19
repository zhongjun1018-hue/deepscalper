"""LightGBM 前瞻预测模型：每个目标一个 LGBMRegressor（平方损失，验证集早停）。

预测目标为窗口缓存的 5 个前瞻标签（data_provider/windows.py TARGET_NAMES）。
fit/predict 输入为 (n, 48) 二维数组：47 维窗口特征（NaN 由 LightGBM 原生处理）
加末列 symbol_id（标的在运行标的集合中的索引，分类特征，见 data/features.md §3.7）。
"""

import glob
import json
import os

import numpy as np

from data_provider.windows import FEATURE_NAMES, TARGET_NAMES

OBJECTIVE = "regression"
SYMBOL_FEATURE = "symbol_id"

# LightGBM 默认超参
DEFAULT_MODEL_KWARGS = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}


class Model:
    """每目标一个 LGBMRegressor；输入末列为 symbol_id 分类特征。"""

    def __init__(self, seed=2021, data_hash="", model_kwargs=None):
        self.seed = seed
        self.data_hash = data_hash
        self.feature_names = list(FEATURE_NAMES) + [SYMBOL_FEATURE]
        self.label_names = list(TARGET_NAMES)
        kwargs = dict(DEFAULT_MODEL_KWARGS)
        kwargs.update(model_kwargs or {})
        conflicts = {name: kwargs[name] for name in ("objective", "metric")
                     if name in kwargs}
        if conflicts:
            raise ValueError(f"训练目标参数不允许覆盖: {conflicts}")
        self.early_stopping_rounds = kwargs.pop("early_stopping_rounds")
        self.params = {"n_jobs": -1, "verbose": -1,
                       "objective": OBJECTIVE, "metric": "l2", **kwargs}
        self.params.setdefault("random_state", seed)
        self.boosters = []

    def _check_features(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"特征形状须为 (n, {len(self.feature_names)})")
        return values

    def fit(self, features, targets, eval_set):
        """池化训练集拟合、验证集逐目标早停；eval_set 为 (val_x, val_y)。"""
        import lightgbm as lgb

        train_x = self._check_features(features)
        train_y = np.asarray(targets, dtype=np.float32)
        val_x = self._check_features(eval_set[0])
        val_y = np.asarray(eval_set[1], dtype=np.float32)
        self.boosters = []
        for column in range(len(self.label_names)):
            regressor = lgb.LGBMRegressor(**self.params)
            regressor.fit(
                train_x,
                np.ascontiguousarray(train_y[:, column]),
                eval_X=val_x,
                eval_y=np.ascontiguousarray(val_y[:, column]),
                categorical_feature=[len(self.feature_names) - 1],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            self.boosters.append(regressor.booster_)
        return self

    def predict(self, features):
        """(n, num_features) → (n, num_targets) float32。"""
        if len(self.boosters) != len(self.label_names):
            raise RuntimeError("模型尚未训练或加载")
        values = self._check_features(features)
        prediction = np.column_stack([booster.predict(values) for booster in self.boosters])
        return prediction.astype(np.float32)

    def _metadata(self):
        return {
            "model": "LightGBM",
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "data_hash": self.data_hash,
            "seed": self.seed,
        }

    def save(self, path):
        """booster_{i}.txt × num_targets + model.json（元数据含 data_hash、特征/目标名、seed）。"""
        if len(self.boosters) != len(self.label_names):
            raise RuntimeError("模型尚未训练或加载")
        os.makedirs(path, exist_ok=True)
        for stale in glob.glob(os.path.join(path, "booster_*.txt")):
            os.remove(stale)
        for column, booster in enumerate(self.boosters):
            booster.save_model(os.path.join(path, f"booster_{column}.txt"))
        with open(os.path.join(path, "model.json"), "w") as file:
            json.dump(self._metadata(), file, indent=2)

    def load(self, path):
        """严格校验 model.json 元数据后加载全部 booster。"""
        import lightgbm as lgb

        with open(os.path.join(path, "model.json")) as file:
            metadata = json.load(file)
        if metadata != self._metadata():
            raise ValueError(f"模型检查点 [{path}] 与当前配置不一致")
        paths = [os.path.join(path, f"booster_{column}.txt")
                 for column in range(len(self.label_names))]
        if any(not os.path.exists(model_path) for model_path in paths):
            raise FileNotFoundError(f"模型检查点 [{path}] 不完整")
        self.boosters = [lgb.Booster(model_file=model_path) for model_path in paths]
        return self
