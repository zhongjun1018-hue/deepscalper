"""模型：多模态市场编码器 + 标的 embedding + dueling Q 网络。

  (a) 微观编码器：LOB 序列与私有状态序列各经一层 LSTM，取末隐状态拼接；
  (b) 宏观编码器：bar 级相对指标与窗口统计、LightGBM 预测向量经 MLP；
  (c) 标的标识：symbol_id 经 embedding 拼接进市场嵌入（统一训练时区分标的）；
  (d) 动作头：共享状态价值 V 与半宽优势聚合为 Q 值；数量档收缩为单档时仅有
      半宽分支（输出 7+1），恢复多档配置时按分支结构扩展数量优势头（design 5.1）。
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
        self.symbol_emb = nn.Embedding(cfg.n_symbols, cfg.symbol_embed_dim)
        self.embed_dim = 2 * h + cfg.macro_hidden + cfg.symbol_embed_dim

    def forward(
        self,
        micro_lob: torch.Tensor,
        private: torch.Tensor,
        macro: torch.Tensor,
        symbol: torch.Tensor,
    ) -> torch.Tensor:
        hb = self.lob_lstm(micro_lob)[1][0][-1]
        hz = self.priv_lstm(private)[1][0][-1]
        ea = self.macro_mlp(macro)
        return torch.cat([hb, hz, ea, self.symbol_emb(symbol)], dim=-1)


class BranchQNetwork(nn.Module):
    """dueling Q 网络：共享状态价值 + 半宽优势；数量多档时另加数量优势分支。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = MarketEncoder(cfg)
        d = self.encoder.embed_dim
        self.trunk = nn.Sequential(nn.Linear(d, cfg.trunk_hidden), nn.ReLU())
        self.value_head = nn.Linear(cfg.trunk_hidden, 1)
        branches = (cfg.n_width,) if cfg.n_size == 1 else (cfg.n_width, cfg.n_size)
        self.branch_heads = nn.ModuleList(
            nn.Linear(cfg.trunk_hidden, n) for n in branches
        )

    def forward(
        self,
        micro_lob: torch.Tensor,
        private: torch.Tensor,
        macro: torch.Tensor,
        symbol: torch.Tensor,
    ) -> list[torch.Tensor]:
        e = self.encoder(micro_lob, private, macro, symbol)
        s = self.trunk(e)
        v = self.value_head(s)
        advantages = [head(s) for head in self.branch_heads]
        return [v + (a - a.mean(dim=-1, keepdim=True)) for a in advantages]


def to_batch(
    obs_list, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 Observation 列表堆叠为网络输入张量 (micro_lob, private, macro, symbol)。"""
    return (
        torch.as_tensor(np.stack([o.micro_lob for o in obs_list])).to(device),
        torch.as_tensor(np.stack([o.private for o in obs_list])).to(device),
        torch.as_tensor(np.stack([o.macro for o in obs_list])).to(device),
        torch.as_tensor([o.symbol_id for o in obs_list], dtype=torch.long).to(device),
    )
