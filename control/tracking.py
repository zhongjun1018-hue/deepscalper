"""wandb 实验跟踪：RL 作业的训练曲线与测试集诊断（design 7.5）。

一个作业对应一个 run：在每个验证评估点记录训练奖励、Q 损失与验证指标（逐标的
SR 的等权聚合），训练结束后记录测试集的四指标（全体等权）、日均买卖笔数与日均
闭环率，并以表格保留逐标的、逐日的超额收益与闭环率。曲线横轴为累计梯度更新次数，
epoch 一并作为指标记录，可在 UI 中改用其为横轴。
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
    "val_TR": "val/TR",
    "val_SR": "val/SR",
    "val_SR_window": "val/SR_window",   # 选模判据
}
TEST_DIAGNOSTICS = ("n_buys", "n_sells", "closure_rate")


class Tracker:
    """单个 RL 作业的 wandb run；run 名与结果文件同名，按方法分类。"""

    def __init__(self, cfg: Config, name: str, job: dict):
        self.run = wandb.init(
            project=cfg.wandb_project,
            mode=cfg.wandb_mode,
            name=name,
            job_type=job["method"],
            config={**dataclasses.asdict(cfg), **job},
        )

    def log_eval(self, record: dict) -> None:
        """记录一个验证评估点，横轴为累计梯度更新次数。"""
        self.run.log({v: record[k] for k, v in EVAL_SERIES.items()}, step=record["updates"])

    def log_test(self, payload: dict) -> None:
        """记录测试集评估：全体等权指标进 summary，逐标的、逐日序列进表格。"""
        diagnostics = payload["diagnostics"]
        self.run.summary.update(
            {f"test/{k}": payload[k] for k in ("TR", "SR", "CR", "SoR")}
            | {f"test/{k}": diagnostics[k] for k in TEST_DIAGNOSTICS}
        )
        rows = [[symbol, day, ret, closure]
                for symbol, entry in payload["per_symbol"].items()
                for day, (ret, closure) in enumerate(
                    zip(entry["daily_returns"], entry["daily_closure_rate"]))]
        self.run.log({"test/daily": wandb.Table(
            columns=["symbol", "day", "excess_return", "closure_rate"], data=rows)})

    def finish(self) -> None:
        self.run.finish()
