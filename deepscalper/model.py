"""模型：多模态市场编码器 + Branching Dueling Q-Network + 波动率辅助头。

对应论文图 3：
  (a) 微观编码器：LOB 序列与私有状态序列各经一层 LSTM，取末隐状态拼接；
  (b) 宏观编码器：OHLCV+技术指标向量经 MLP；
  (c) 风险辅助任务：市场嵌入经单层 MLP 预测未来波动率；
  (d) 动作分支：共享状态价值 V 与价格 / 数量两支优势函数聚合为 Q 值。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .features import MACRO_DIM, MICRO_DIM, PRIVATE_DIM


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
    """BDQ：共享状态价值 + 价格 / 数量双优势分支，附波动率预测头。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = MarketEncoder(cfg)
        d = self.encoder.embed_dim
        self.trunk = nn.Sequential(nn.Linear(d, cfg.trunk_hidden), nn.ReLU())
        self.value_head = nn.Linear(cfg.trunk_hidden, 1)
        self.price_head = nn.Linear(cfg.trunk_hidden, cfg.n_price)
        self.qty_head = nn.Linear(cfg.trunk_hidden, cfg.n_quantity)
        self.vol_head = nn.Linear(d, 1)

    def forward(
        self, micro_lob: torch.Tensor, private: torch.Tensor, macro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.encoder(micro_lob, private, macro)
        s = self.trunk(e)
        v = self.value_head(s)
        adv_p = self.price_head(s)
        adv_q = self.qty_head(s)
        q_p = v + (adv_p - adv_p.mean(dim=-1, keepdim=True))
        q_q = v + (adv_q - adv_q.mean(dim=-1, keepdim=True))
        return q_p, q_q, self.vol_head(e).squeeze(-1)


def to_batch(obs_list, device: torch.device) -> tuple[torch.Tensor, ...]:
    """将 Observation 列表堆叠为网络输入张量。"""
    micro_lob = torch.as_tensor(np.stack([o.micro_lob for o in obs_list]))
    private = torch.as_tensor(np.stack([o.private for o in obs_list]))
    macro = torch.as_tensor(np.stack([o.macro for o in obs_list]))
    return (
        micro_lob.to(device),
        private.to(device),
        macro.to(device),
    )
