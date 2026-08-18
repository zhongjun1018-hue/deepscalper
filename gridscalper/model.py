"""模型：多模态市场编码器 + Branching Dueling Q-Network + 波动率辅助头。

  (a) 微观编码器：LOB 序列与私有状态序列各经一层 LSTM，取末隐状态拼接；
  (b) 宏观编码器：OHLCV+技术指标向量经 MLP；
  (c) 风险辅助任务：市场嵌入经单层 MLP 预测未来波动率；
  (d) 动作分支：共享状态价值 V 与半宽 / 数量两支优势函数聚合为 Q 值。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .features import MACRO_DIM, MICRO_DIM, PRIVATE_DIM


def resolve_device(cfg: Config) -> torch.device:
    """解析 cfg.device："auto" 时有 CUDA 用 CUDA，否则用 CPU。"""
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


class MarketEncoder(nn.Module):
    """微观 + 宏观双路编码，输出市场嵌入 e_t。"""

    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.hidden_size
        self.lob_lstm = nn.LSTM(MICRO_DIM, h, batch_first=True)
        self.priv_lstm = nn.LSTM(PRIVATE_DIM, h, batch_first=True)
        self.macro_mlp = nn.Sequential(
            nn.Linear(MACRO_DIM, cfg.macro_hidden),
            nn.ReLU(),
            nn.Linear(cfg.macro_hidden, cfg.macro_hidden),
            nn.ReLU(),
        )
        self.embed_dim = 2 * h + cfg.macro_hidden

    def forward(
        self, micro_lob: torch.Tensor, private: torch.Tensor, macro: torch.Tensor
    ) -> torch.Tensor:
        hb = self.lob_lstm(micro_lob)[1][0][-1]
        hz = self.priv_lstm(private)[1][0][-1]
        ea = self.macro_mlp(macro)
        return torch.cat([hb, hz, ea], dim=-1)


class BDQNetwork(nn.Module):
    """BDQ：共享状态价值 + 半宽 / 数量两支优势，附波动率预测头。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = MarketEncoder(cfg)
        d = self.encoder.embed_dim
        self.trunk = nn.Sequential(nn.Linear(d, cfg.trunk_hidden), nn.ReLU())
        self.value_head = nn.Linear(cfg.trunk_hidden, 1)
        self.branch_heads = nn.ModuleList(
            nn.Linear(cfg.trunk_hidden, n) for n in (cfg.n_width, cfg.n_size)
        )
        self.vol_head = nn.Linear(d, 1)

    def forward(
        self, micro_lob: torch.Tensor, private: torch.Tensor, macro: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        e = self.encoder(micro_lob, private, macro)
        s = self.trunk(e)
        v = self.value_head(s)
        advantages = [head(s) for head in self.branch_heads]
        q = [v + (a - a.mean(dim=-1, keepdim=True)) for a in advantages]
        return q, self.vol_head(e).squeeze(-1)


def to_batch(
    obs_list, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 Observation 列表堆叠为网络输入张量 (micro_lob, private, macro)。"""
    return (
        torch.as_tensor(np.stack([o.micro_lob for o in obs_list])).to(device),
        torch.as_tensor(np.stack([o.private for o in obs_list])).to(device),
        torch.as_tensor(np.stack([o.macro for o in obs_list])).to(device),
    )
