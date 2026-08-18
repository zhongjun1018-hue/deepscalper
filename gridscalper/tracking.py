"""wandb 实验跟踪：RL 作业的训练曲线与测试集诊断（design 7.5）。

一个作业对应一个 run：在每个验证评估点记录训练奖励、两段损失与验证指标，训练结束后
记录测试集的四指标、日均买卖笔数与日均闭环率，并以表格保留逐日超额收益与闭环率。
曲线横轴为累计梯度更新次数，epoch 一并作为指标记录，可在 UI 中改用其为横轴。
Config.wandb_mode = "disabled" 时 wandb 自身退化为空操作，无需在调用处分支。
"""

from __future__ import annotations

import dataclasses

import wandb

from .config import Config

# 训练日志字段 → wandb 曲线名
EVAL_SERIES = {
    "epoch": "epoch",
    "train_reward": "train/reward",
    "q_loss": "train/q_loss",
    "vol_loss": "train/vol_loss",
    "val_TR": "val/TR",
    "val_SR": "val/SR",
    "val_SR_window": "val/SR_window",   # 选模判据
}
TEST_DIAGNOSTICS = ("n_buys", "n_sells", "closure_rate")


class Tracker:
    """单个 RL 作业的 wandb run；run 名与结果文件同名，按标的分组、按方法分类。"""

    def __init__(self, cfg: Config, name: str, job: dict):
        self.run = wandb.init(
            project=cfg.wandb_project,
            mode=cfg.wandb_mode,
            name=name,
            group=job["symbol"],
            job_type=job["method"],
            config={**dataclasses.asdict(cfg), **job},
        )

    def log_eval(self, record: dict) -> None:
        """记录一个验证评估点，横轴为累计梯度更新次数。"""
        self.run.log({v: record[k] for k, v in EVAL_SERIES.items()}, step=record["updates"])

    def log_test(self, metrics: dict) -> None:
        """记录测试集评估：汇总指标进 summary，逐日序列进表格。"""
        diagnostics = metrics["diagnostics"]
        self.run.summary.update(
            {f"test/{k}": metrics[k] for k in ("TR", "SR", "CR", "SoR")}
            | {f"test/{k}": diagnostics[k] for k in TEST_DIAGNOSTICS}
        )
        self.run.log({"test/daily": wandb.Table(
            columns=["day", "excess_return", "closure_rate"],
            data=[[i, r, c] for i, (r, c) in enumerate(
                zip(metrics["daily_returns"], metrics["daily_closure_rate"]))],
        )})

    def finish(self) -> None:
        self.run.finish()
