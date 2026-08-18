"""汇总实验结果：输出测试集汇总表，并按验证集 SR 选定超参 (w, λ)。

超参选优只用验证集，且在全部标的上聚合后取同一档：λ 与 w 都是无量纲的偏好参数，
按单只标的分别选取会使其失去跨标的可比性，也更容易过拟合到某段行情（design 7.1）。
"""

from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

METHODS = ["HOLD", "OPEN", "SCAN", "GRID-FW", "GRID-FT", "GRID-FS", "GRID-NH", "GRID-NA", "GRID"]
METRICS = ["TR", "SR", "CR", "SoR"]
RULE_METHODS = {"HOLD", "OPEN", "SCAN"}


def load_rows(result_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({"symbol": r["symbol"], "method": r["method"], "seed": r["seed"],
                     "w": r["hindsight_weight"], "lam": r["inventory_lambda"],
                     "val_SR": r.get("train_log", {}).get("best_val_SR"),
                     **{k: r[k] for k in METRICS}})
    return pd.DataFrame(rows)


def test_summary(df: pd.DataFrame) -> pd.DataFrame:
    """汇总已经锁定超参的测试集指标：多种子给出均值 ± 标准差。"""
    out = []
    for keys, g in df.groupby(["symbol", "method", "w", "lam"], dropna=False):
        row = dict(zip(("symbol", "method", "w", "lam"), keys)) | {"n_runs": len(g)}
        for k in METRICS:
            v = g[k].to_numpy()
            row[k] = f"{v.mean():.4f} ± {v.std():.4f}" if len(g) > 1 else f"{v.mean():.4f}"
        out.append(row)
    summary = pd.DataFrame(out)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    summary = summary.sort_values(["symbol", "method", "w", "lam"])
    for col in ("w", "lam"):
        summary[col] = summary[col].map(lambda x: "" if pd.isna(x) else f"{x:g}")
    return summary


def select_hyperparams(df: pd.DataFrame) -> pd.DataFrame:
    """按验证集 SR 在全部标的与种子上取均值，逐方法给出各超参格点的排名。"""
    rl = df[df["val_SR"].notna()]
    if rl.empty:
        return pd.DataFrame()
    sel = (rl.groupby(["method", "w", "lam"], dropna=False)["val_SR"]
             .agg(val_SR="mean", n_runs="size").reset_index())
    return sel.sort_values(["method", "val_SR"], ascending=[True, False])


def selected_test_rows(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """保留规则基线与各 RL 方法验证集排名第一的超参档。"""
    keep = df["method"].isin(RULE_METHODS)
    for method, group in selection.groupby("method", dropna=False):
        best = group.loc[group["val_SR"].idxmax()]
        same_w = df["w"].isna() if pd.isna(best["w"]) else df["w"].eq(best["w"])
        same_lam = df["lam"].isna() if pd.isna(best["lam"]) else df["lam"].eq(best["lam"])
        keep |= df["method"].eq(method) & same_w & same_lam
    return df[keep]


def main(result_dir: str = "results") -> None:
    df = load_rows(result_dir)
    if df.empty:
        print(f"未找到结果文件：{result_dir}")
        return

    sel = select_hyperparams(df)
    summary = test_summary(selected_test_rows(df, sel))
    out_path = os.path.join(result_dir, "summary.csv")
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))

    if not sel.empty:
        print("\n超参选优（验证集 SR，跨标的与种子取均值）：")
        print(sel.to_string(index=False))
        full = sel[sel["method"] == "GRID"]
        if not full.empty:
            best = full.loc[full["val_SR"].idxmax()]
            print(f"\nGRID 选定：w={best['w']:g}  λ={best['lam']:g}  (val_SR={best['val_SR']:.3f})")
    print(f"\n已保存：{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
