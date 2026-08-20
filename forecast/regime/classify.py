"""模式识别模型：LightGBM 二分类 P(网格不利模式 | 47 维窗口特征 + symbol_id)。

识别质量以 AUC / AP 对着事后标签评估，与策略盈亏解耦。行口径与标签一致：
分钟锚点行、特征任一维有限、标签可判定（>= 0）。推理同在锚点行上进行，
门控使用方按锚点前向填充到 tick（regime.data.expand_minutes，保持因果性）。
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from data_provider.windows import FEATURE_NAMES

from forecast.regime.data import SymbolBank

SYMBOL_FEATURE = "symbol_id"


class Classifier:
    """单 booster 二分类器；输入 (n, 48)，末列为 symbol_id 分类特征。"""

    def __init__(self, seed=2021, data_hash="", model_kwargs=None):
        self.seed = seed
        self.data_hash = data_hash
        self.feature_names = list(FEATURE_NAMES) + [SYMBOL_FEATURE]
        kwargs = dict(model_kwargs or {})
        self.early_stopping_rounds = kwargs.pop("early_stopping_rounds")
        self.params = {"n_jobs": -1, "verbose": -1, "objective": "binary",
                       "metric": "auc", "random_state": seed, **kwargs}
        self.booster = None

    def _check_features(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"特征形状须为 (n, {len(self.feature_names)})")
        return values

    def fit(self, features, labels, eval_set):
        """池化训练集拟合、验证集 AUC 早停；eval_set 为 (val_x, val_y)。"""
        import lightgbm as lgb

        classifier = lgb.LGBMClassifier(**self.params)
        classifier.fit(
            self._check_features(features), np.asarray(labels),
            eval_X=self._check_features(eval_set[0]),
            eval_y=np.asarray(eval_set[1]),
            categorical_feature=[len(self.feature_names) - 1],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                       lgb.log_evaluation(0)])
        self.booster = classifier.booster_
        return self

    def predict(self, features) -> np.ndarray:
        """(n, 48) → (n,) 不利模式概率。"""
        if self.booster is None:
            raise RuntimeError("模型尚未训练或加载")
        return np.asarray(self.booster.predict(self._check_features(features)))

    def _metadata(self):
        return {"model": "LightGBM", "objective": "binary",
                "feature_names": self.feature_names,
                "data_hash": self.data_hash, "seed": self.seed}

    def save(self, path):
        """booster.txt + model.json（元数据含 data_hash、特征名、seed）。"""
        if self.booster is None:
            raise RuntimeError("模型尚未训练或加载")
        os.makedirs(path, exist_ok=True)
        for stale in glob.glob(os.path.join(path, "booster*.txt")):
            os.remove(stale)
        self.booster.save_model(os.path.join(path, "booster.txt"))
        with open(os.path.join(path, "model.json"), "w") as file:
            json.dump(self._metadata(), file, indent=2)

    def load(self, path):
        """严格校验 model.json 元数据后加载 booster。"""
        import lightgbm as lgb

        with open(os.path.join(path, "model.json")) as file:
            metadata = json.load(file)
        if metadata != self._metadata():
            raise ValueError(f"模型检查点 [{path}] 与当前配置不一致")
        self.booster = lgb.Booster(
            model_file=os.path.join(path, "booster.txt"))
        return self


def symbol_rows(bank: SymbolBank, labels: np.ndarray, flag: str,
                symbol_id: int) -> tuple:
    """单标的某切分段的 (x (n,48) f32, y (n,) int8) 分钟锚点行。"""
    xs, ys = [], []
    for i in bank.day_indices(flag):
        feats = bank.features[i]
        label = labels[i]
        ok = (label >= 0) & np.isfinite(feats).any(axis=1)
        xs.append(feats[ok])
        ys.append(label[ok])
    x = (np.concatenate(xs) if xs
         else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32))
    x = np.column_stack([x, np.full(len(x), symbol_id, dtype=np.float32)])
    return x.astype(np.float32), (np.concatenate(ys) if ys
                                  else np.empty(0, dtype=np.int8))


def pooled_rows(banks: dict, labels: dict, flag: str) -> tuple:
    """跨标的池化 (x, y)；symbol_id 为排序后标的集合中的索引。"""
    rows = [symbol_rows(banks[symbol], labels[symbol], flag, index)
            for index, symbol in enumerate(sorted(banks))]
    return (np.concatenate([x for x, _ in rows]),
            np.concatenate([y for _, y in rows]))


def day_prob(classifier: Classifier, bank: SymbolBank, day_index: int,
             symbol_id: int) -> np.ndarray:
    """单日逐分钟概率 (M,)：锚点行上推理；无特征的分钟记 NaN。"""
    feats = bank.features[day_index]
    ok = np.isfinite(feats).any(axis=1)
    prob = np.full(len(feats), np.nan)
    if ok.any():
        x = np.column_stack([feats[ok],
                             np.full(ok.sum(), symbol_id, dtype=np.float32)])
        prob[ok] = classifier.predict(x)
    return prob


def calibrate_threshold(probabilities, labels) -> float:
    """率配平阈值：验证段预测判正率 = 真值不利占比（不含测试段与盈亏信息）。"""
    return float(np.quantile(probabilities, 1.0 - float(np.mean(labels))))
