"""汇总实验结果：输出各折的测试集汇总表与行情状态，并按验证集 SR 选定超参 (w, λ)。

结果按折组织在 results/<symbol>/fold_<折>/ 下，汇总全部折。超参选优只用验证集，
在全部标的与种子上聚合后逐折取同一档：λ 与 w 都是无量纲的偏好参数，
按单只标的分别选取会使其失去跨标的可比性；而跨折聚合会把折 i 的测试窗口
经折 i+1 的验证窗口带入折 i 的选参（design 7.1）。
测试指标按折分行报告：各折测试窗口的市场状态不同，跨折平均会掩盖状态差异。
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

METHODS = ["HOLD", "OPEN", "SCAN", "GRID-FW", "GRID-FT", "GRID-FS", "GRID-NH", "GRID-NA", "GRID"]
METRICS = ["TR", "SR", "CR", "SoR"]
RULE_METHODS = {"HOLD", "OPEN", "SCAN"}


def load_rows(result_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*", "fold_*", "*.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        rows.append({"symbol": r["symbol"], "fold": r["fold"], "method": r["method"],
                     "seed": r["seed"], "splits": r["splits"],
                     "w": r["hindsight_weight"], "lam": r["inventory_lambda"],
                     "val_SR": r.get("train_log", {}).get("best_val_SR"),
                     **{k: r[k] for k in METRICS}})
    return pd.DataFrame(rows)


def test_summary(df: pd.DataFrame) -> pd.DataFrame:
    """汇总已经锁定超参的测试集指标：多种子给出均值 ± 标准差。"""
    out = []
    for keys, g in df.groupby(["symbol", "fold", "method", "w", "lam"], dropna=False):
        row = dict(zip(("symbol", "fold", "method", "w", "lam"), keys)) | {"n_runs": len(g)}
        for k in METRICS:
            v = g[k].to_numpy()
            row[k] = f"{v.mean():.4f} ± {v.std():.4f}" if len(g) > 1 else f"{v.mean():.4f}"
        out.append(row)
    summary = pd.DataFrame(out)
    summary["method"] = pd.Categorical(summary["method"], METHODS, ordered=True)
    summary = summary.sort_values(["symbol", "fold", "method", "w", "lam"])
    for col in ("w", "lam"):
        summary[col] = summary[col].map(lambda x: "" if pd.isna(x) else f"{x:g}")
    return summary


def regime_table(df: pd.DataFrame) -> pd.DataFrame:
    """各折训练 / 验证 / 测试段的行情状态：日内漂移、上涨日占比与相对波动。"""
    rows = [{"symbol": r.symbol, "fold": r.fold, "split": split, **stats}
            for r in df.drop_duplicates(["symbol", "fold"]).itertuples()
            for split, stats in r.splits.items()]
    return pd.DataFrame(rows).sort_values(["symbol", "fold"], kind="stable").round(4)


def select_hyperparams(df: pd.DataFrame) -> pd.DataFrame:
    """按验证集 SR 在全部标的与种子上取均值，逐折、逐方法给出各超参格点的排名。"""
    rl = df[df["val_SR"].notna()]
    if rl.empty:
        return pd.DataFrame()
    sel = (rl.groupby(["fold", "method", "w", "lam"], dropna=False)["val_SR"]
             .agg(val_SR="mean", n_runs="size").reset_index())
    return sel.sort_values(["fold", "method", "val_SR"], ascending=[True, True, False])


def selected_test_rows(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """保留规则基线，与各折各 RL 方法验证集排名第一的超参档。"""
    keep = df["method"].isin(RULE_METHODS)
    for (fold, method), group in selection.groupby(["fold", "method"], dropna=False):
        best = group.loc[group["val_SR"].idxmax()]
        same_w = df["w"].isna() if pd.isna(best["w"]) else df["w"].eq(best["w"])
        same_lam = df["lam"].isna() if pd.isna(best["lam"]) else df["lam"].eq(best["lam"])
        keep |= df["fold"].eq(fold) & df["method"].eq(method) & same_w & same_lam
    return df[keep]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results", help="结果根目录（<symbol>/fold_<折>/ 结构）")
    args = p.parse_args()

    df = load_rows(args.results)
    if df.empty:
        print(f"未找到结果文件：{args.results}")
        return

    sel = select_hyperparams(df)
    summary = test_summary(selected_test_rows(df, sel))
    out_path = os.path.join(args.results, "summary.csv")
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))

    print("\n各折行情状态（解读测试指标的前提）：")
    print(regime_table(df).to_string(index=False))

    if not sel.empty:
        print("\n超参选优（验证集 SR，逐折、跨标的与种子取均值）：")
        print(sel.to_string(index=False))
        for fold, g in sel[sel["method"] == "GRID"].groupby("fold"):
            best = g.loc[g["val_SR"].idxmax()]
            print(f"GRID 折 {fold} 选定：w={best['w']:g}  λ={best['lam']:g}  "
                  f"(val_SR={best['val_SR']:.3f})")
    print(f"\n已保存：{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
