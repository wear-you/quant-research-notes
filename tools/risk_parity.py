#!/usr/bin/env python3
"""risk_parity.py — 用「风险预测」而不是「收益预测」构建组合。

逻辑起点：
  项目里唯一反复被验证的能力是**预测波动率**（沪深300 R² 0.327，MAE 4/5 改善），
  而收益方向预测全部失败（AUC 0.51，IC 不显著）。
  所以唯一自洽的方向是：不去猜收益，只按风险分配仓位，靠分散化拿风险溢价。

三个变体（同一池子、同一区间、同一成本）：
  A 等权买入持有        —— 基准
  B 逆波动率（风险平价） —— 按 1/vol 分配，组合目标波动 10%
  C 逆波动率 + 趋势过滤  —— 额外剔除价格低于 MA100 的资产

严格性：
  - 每 20 个交易日调仓；权重只用 t 日及之前的数据
  - 权重之和 <= 1（不加杠杆），剩余部分按 1.5% 年化计息
  - 成本 = 权重变动绝对值 × 单边 8bp（A 股 ETF/指数现实区间）
  - 多资产按日期对齐（market.py），禁止索引对齐
"""
import csv
import math
import os
import statistics
import sys

import market

FIN = os.path.expanduser("~/finance")
TD = 244
TARGET = float(os.environ.get("RP_TARGET", 0.10))
CASH = 0.015
COST = float(os.environ.get("RP_COST", 0.0008))
REBAL = int(os.environ.get("RP_REBAL", 20))
VOLW = 60
MAW = int(os.environ.get("RP_MAW", 100))
MIN_BARS = 250
MAX_W = float(os.environ.get("RP_MAXW", 0.35))
START = next((a for a in sys.argv[1:] if a[:2] == "20" and "-" in a), "2016-01-01")

EXCLUDE = {"csi_all", "csi_div", "nasdaq_etf"}
CRYPTO = {"btc", "eth", "sol", "xrp", "ada", "doge", "avax", "dot", "link",
          "ltc", "bch", "bnb", "trx"}
if "--nocrypto" in sys.argv:
    EXCLUDE = EXCLUDE | CRYPTO
if "--onlycrypto" in sys.argv:
    EXCLUDE = {n for n in ("ada avax bch bnb bond10y bond_index btc china_internet "
                           "chinext chinext50 csi100 csi1000 csi300 csi500 csi_consume "
                           "doge dot eth gold hscei hsi hstech link ltc moutai nasdaq_etf "
                           "sec_etf sol sse50 sse_comp star50 szse_comp trx xrp").split()
               if n not in CRYPTO}


def load_all():
    names = []
    for f in sorted(os.listdir(os.path.join(FIN, "data"))):
        if not f.endswith(".csv"):
            continue
        n = f[:-4]
        if n in EXCLUDE:
            continue
        s = market.load_series(n)
        if len(s) >= MIN_BARS:
            names.append(n)
    return names


def panel(names):
    """并集日历 + 每个资产自己的价格序列（允许不同起始日）。"""
    ref = market.load_series("csi300")
    dates = sorted(ref)
    px = {n: market.load_series(n) for n in names}
    # 只保留 2016 之后、且在参考日历上的日期
    dates = [d for d in dates if d >= START]
    for n in names:
        px[n] = {d: v for d, v in px[n].items() if d >= "2010-01-01"}
    return dates, px


def rets_from(px, n):
    ds = sorted(px[n])
    return {ds[i]: px[n][ds[i]] / px[n][ds[i - 1]] - 1 for i in range(1, len(ds))}, ds


def main():
    names = load_all()
    dates, px = panel(names)
    print(f"资产 {len(names)} 个，日历 {len(dates)} 天：{dates[0]} → {dates[-1]}")

    hist = {n: sorted(px[n]) for n in names}
    idx = {n: {d: i for i, d in enumerate(hist[n])} for n in names}
    rets = {}
    for n in names:
        h = hist[n]
        rets[n] = {h[i]: px[n][h[i]] / px[n][h[i - 1]] - 1 for i in range(1, len(h))}

    def vol_of(n, d, w=VOLW):
        i = idx[n].get(d)
        if i is None or i < w:
            return None
        h = hist[n]
        r = [rets[n].get(h[j]) for j in range(i - w + 1, i + 1)]
        r = [x for x in r if x is not None]
        if len(r) < w * 0.8:
            return None
        return statistics.pstdev(r) * TD ** 0.5

    def above_ma(n, d):
        i = idx[n].get(d)
        if i is None or i < MAW:
            return False
        h = hist[n]
        return px[n][d] > sum(px[n][h[j]] for j in range(i - MAW + 1, i + 1)) / MAW

    strategies = {"EQW": [], "RP": [], "RP+TREND": []}
    weights = {"RP": {}, "RP+TREND": {}, "EQW": {}}
    turnover = {"RP": 0.0, "RP+TREND": 0.0}
    last_rebal = -10 ** 9

    for k, d in enumerate(dates):
        if k == 0:
            continue
        # 调仓
        if k - last_rebal >= REBAL:
            last_rebal = k
            avail = [n for n in names if vol_of(n, d)]
            for tag, use_trend in (("RP", False), ("RP+TREND", True)):
                pool = [n for n in avail if (above_ma(n, d) if use_trend else True)]
                new = {}
                if pool:
                    inv = {n: 1.0 / max(vol_of(n, d), 1e-4) for n in pool}
                    tot = sum(inv.values())
                    w = {n: inv[n] / tot for n in pool}
                    w = {n: min(v, MAX_W) for n, v in w.items()}
                    s = sum(w.values())
                    w = {n: v / s for n, v in w.items()}
                    # 组合波动（用对角近似 + 平均相关 0.25 的保守估计）
                    sig = math.sqrt(sum((w[n] * vol_of(n, d)) ** 2 for n in w)
                                    + 0.25 * (sum(w[n] * vol_of(n, d) for n in w) ** 2
                                              - sum((w[n] * vol_of(n, d)) ** 2 for n in w)))
                    scale = min(1.0, TARGET / sig) if sig > 0 else 0.0
                    new = {n: v * scale for n, v in w.items()}
                old = weights[tag]
                turnover[tag] += sum(abs(new.get(n, 0) - old.get(n, 0))
                                     for n in set(new) | set(old))
                weights[tag] = new
            if not weights["EQW"]:
                pool = [n for n in avail]
                weights["EQW"] = {n: 1.0 / len(pool) for n in pool} if pool else {}

        # 当日收益
        for tag in strategies:
            w = weights[tag]
            r = 0.0
            for n, wt in w.items():
                rr = rets[n].get(d)
                if rr is not None:
                    r += wt * rr
            invested = sum(w.values())
            r += (1 - invested) * (CASH / TD)
            strategies[tag].append(r)

    def perf(r):
        eq, peak, mdd = 1.0, 1.0, 0.0
        for v in r:
            eq *= 1 + v
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
        sd = statistics.pstdev(r)
        return {"cagr": eq ** (TD / len(r)) - 1, "vol": sd * TD ** 0.5,
                "sharpe": statistics.mean(r) / sd * TD ** 0.5 if sd else 0,
                "mdd": mdd, "n": len(r)}

    print(f"\n区间 {dates[1]} → {dates[-1]}，{len(strategies['EQW'])} 天")
    print(f"{'策略':10s} {'夏普':>6s} {'年化':>8s} {'波动':>7s} {'回撤':>7s} {'换手/年':>9s}")
    print("-" * 56)
    out = {}
    for tag in ("EQW", "RP", "RP+TREND"):
        p = perf(strategies[tag])
        to = turnover.get(tag, 0.0) / (len(strategies[tag]) / TD)
        out[tag] = p
        print(f"{tag:10s} {p['sharpe']:6.2f} {p['cagr']*100:7.2f}% "
              f"{p['vol']*100:6.2f}% {p['mdd']*100:6.1f}% {to*100:8.0f}%")

    # 扣费后的 RP（成本按换手计）
    for tag in ("RP", "RP+TREND"):
        to = turnover[tag] / (len(strategies[tag]) / TD)
        drag = to * COST * 2
        net = out[tag]["cagr"] - drag
        print(f"\n{tag} 扣费后年化：{net*100:.2f}%（换手 {to*100:.0f}%/年 × 双边 "
              f"{COST*2*1e4:.0f}bp = {drag*100:.2f}pp）")

    print("\n平均仓位：", end="")
    for tag in ("RP", "RP+TREND"):
        w = weights[tag]
        print(f"{tag} 末次 {sum(w.values())*100:.0f}%  ", end="")
    print()

    # 归因：末次权重构成
    for tag in ("RP+TREND",):
        w = weights[tag]
        print(f"\n{tag} 末次持仓（前 10）：")
        for n, v in sorted(w.items(), key=lambda x: -x[1])[:10]:
            print(f"  {n:14s} {v*100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
