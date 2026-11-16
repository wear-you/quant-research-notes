#!/usr/bin/env python3
"""validate.py — 回测验证闸门：CSCV 过拟合概率(PBO) + Deflated Sharpe Ratio(DSR)。

为什么必须做：
  我们此前所有"夏普 X"都没有校正"试了多少次参数"。按 Bailey & López de Prado，
  试 N 次后的最高夏普会系统性偏高——不校正就等于自欺。

只用标准库（statistics.NormalDist 提供逆正态），不依赖 numpy/scipy。

CSCV (Bailey et al. 2017)：
  把 T 期切成 S 个连续块，枚举所有 C(S, S/2) 种"一半做样本内、一半做样本外"的组合；
  每个组合里，样本内最优的配置在样本外排名如何？
  PBO = 该配置样本外排名落到中位数以下的组合占比。PBO > 0.5 → 过拟合。

DSR (Bailey & López de Prado 2014)：
  SR0 = sqrt(V[SR_trials]) * ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e)))
  DSR = Φ( (SR - SR0)·sqrt(T-1) / sqrt(1 - skew·SR + (kurt-1)/4·SR²) )
"""
import csv
import glob
import itertools
import math
import os
import statistics
import sys
from statistics import NormalDist

DATA = os.path.expanduser("~/finance/data")
TD = 244
ND = NormalDist()
GAMMA = 0.5772156649015329

CRYPTO = {"btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "avax",
          "dot", "ltc", "bch", "trx"}

WINDOWS = (100, 150, 200, 250)
TARGETS = (0.10, 0.15, 0.20, 0.30)
VOLW = 60


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return None
    rows = [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]
    return rows if len(rows) > 500 else None


def series_for(px, window, target, cost=0.0008):
    """VOLT 策略的单资产日收益序列。"""
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, len(px))]
    out, prev = [], 0.0
    for i in range(max(window, VOLW), len(px) - 1):
        ma = sum(px[i - window + 1:i + 1]) / window
        if px[i] <= ma:
            w = 0.0
        else:
            sd = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
            w = min(1.0, target / sd) if sd > 0 else 0.0
        pnl = w * (px[i + 1] / px[i] - 1) - abs(w - prev) * cost
        out.append(pnl)
        prev = w
    return out


def sharpe(r):
    if len(r) < 2:
        return 0.0
    sd = statistics.pstdev(r)
    return (statistics.mean(r) / sd) * TD ** 0.5 if sd else 0.0


def build_matrix(names, window, target):
    """把多个资产拼成一个等权组合，返回日收益序列。"""
    per_asset = []
    for n in names:
        rows = load(n)
        if not rows:
            continue
        per_asset.append(series_for([p for _, p in rows], window, target))
    if not per_asset:
        return []
    L = min(len(s) for s in per_asset)
    return [sum(s[i] for s in per_asset) / len(per_asset) for i in range(L)]


def cscv(matrix, S=10):
    """matrix: list of configs, each a list of returns (等长).
    返回 PBO。"""
    T = min(len(m) for m in matrix)
    N = len(matrix)
    block = T // S
    idx = [[i * block for i in range(S)], [(i + 1) * block for i in range(S)]]

    below = 0
    total = 0
    for combo in itertools.combinations(range(S), S // 2):
        is_idx, oos_idx = [], []
        lo = {}
        for b in range(S):
            lo[b] = (b * block, (b + 1) * block)
        for b in combo:
            is_idx.extend(range(*lo[b]))
        for b in range(S):
            if b not in combo:
                oos_idx.extend(range(*lo[b]))
        is_sr = [sharpe([m[i] for i in is_idx]) for m in matrix]
        best = max(range(N), key=lambda k: is_sr[k])
        oos_sr = [sharpe([m[i] for i in oos_idx]) for m in matrix]
        rank = sum(1 for s in oos_sr if s <= oos_sr[best])  # 1 = 最差
        omega = rank / (N + 1)
        total += 1
        if omega <= 0.5:
            below += 1
    return below / total if total else float("nan"), total


def dsr(r, n_trials, sr_variance):
    """r: 日收益序列；n_trials: 试过多少种配置；sr_variance: 各配置夏普的方差（年化口径）。"""
    T = len(r)
    sr_ann = sharpe(r)
    sr = sr_ann / math.sqrt(TD)          # 期内夏普
    if T < 3 or sr_variance <= 0:
        return float("nan"), sr_ann
    sd_sr = math.sqrt(sr_variance) / math.sqrt(TD)
    z1 = ND.inv_cdf(1 - 1.0 / n_trials)
    z2 = ND.inv_cdf(1 - 1.0 / (n_trials * math.e))
    sr0 = sd_sr * ((1 - GAMMA) * z1 + GAMMA * z2)
    m = statistics.mean(r)
    sd = statistics.pstdev(r)
    if sd == 0:
        return float("nan"), sr_ann
    x = [(v - m) / sd for v in r]
    skew = sum(v ** 3 for v in x) / T
    kurt = sum(v ** 4 for v in x) / T
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan"), sr_ann
    z = (sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom)
    return ND.cdf(z), sr_ann


def main():
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.csv"))
                   if os.path.basename(p)[:-4] not in CRYPTO)
    names = [n for n in names if load(n)]
    if "--long" in sys.argv:
        names = [n for n in names if len(load(n)) >= 5000]
        print("【长样本模式】只保留 >=5000 根的资产")
    print(f"资产池 {len(names)} 个：{', '.join(names)}")

    configs, labels = [], []
    for w in WINDOWS:
        for t in TARGETS:
            s = build_matrix(names, w, t)
            if len(s) > 1000:
                configs.append(s)
                labels.append(f"MA{w}/tgt{t:.0%}")

    T = min(len(c) for c in configs)
    configs = [c[:T] for c in configs]
    print(f"配置 {len(configs)} 个，共同窗口 {T} 天\n")

    srs = [sharpe(c) for c in configs]
    for lab, s in sorted(zip(labels, srs), key=lambda x: -x[1]):
        print(f"  {lab:16s} 夏普 {s:5.2f}")
    best_i = max(range(len(srs)), key=lambda i: srs[i])
    best_sr = srs[best_i]
    var_sr = statistics.pvariance(srs) if len(srs) > 1 else 0.0

    print(f"\n最高夏普配置：{labels[best_i]} = {best_sr:.2f}")
    print(f"各配置夏普方差（年化）：{var_sr:.4f}")

    pbo, n_combos = cscv(configs, S=10)
    d, sr_ann = dsr(configs[best_i], n_trials=len(configs), sr_variance=var_sr)

    print(f"\n=== 判决 ===")
    print(f"PBO（过拟合概率）= {pbo:.1%}   （{n_combos} 个组合；>50% 判过拟合）")
    print(f"DSR（校正后为真概率）= {d:.1%}   （原始夏普 {sr_ann:.2f}）")
    verdict = "通过" if (pbo < 0.5 and d > 0.95) else "不通过"
    print(f"结论：{verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
