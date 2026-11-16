#!/usr/bin/env python3
"""verify_holdout.py — 用「日期对齐」重算 holdout，检查之前那个 1.88 是否成立。

可疑点：
  holdout.py 的 portfolio() 按**索引**对齐多个资产的收益序列，
  但这些资产起始日期不同（1990 / 2002 / 2005），索引 i 对应的日历日期不一样。
  把不同日子的收益平均在一起，会人为制造分散化，抬高夏普。

这里改成：只在**所有资产都有数据的共同交易日**上计算，逐日对齐。
"""
import csv
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
TD = 244
CUT = "2016-01-01"
COST = 0.0008
CASH = 0.015
WINDOW, VOLW = 100, 60
TARGET = 0.10


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return {}
    return {r["date"]: float(r["close"]) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0}


def main():
    names = ["bond_index", "csi100", "csi300", "csi500", "sse50", "sse_comp", "szse_comp"]
    series = {n: load(n) for n in names}
    series = {n: s for n, s in series.items() if s}

    common = set.intersection(*[set(s) for s in series.values()])
    dates = sorted(d for d in common if d >= CUT)
    print(f"共同交易日 {len(dates)} 天：{dates[0]} → {dates[-1]}")

    # 每个资产：在共同日期上取价格，用其自身历史算信号
    px = {n: [series[n][d] for d in dates] for n in series}
    # 为了算 MA100/vol60，需要把 cut 之前的历史也接上
    full = {}
    for n in series:
        ds = sorted(series[n])
        full[n] = (ds, [series[n][d] for d in ds])

    port, bah = [], []
    prev_w = {n: 0.0 for n in series}
    for k, d in enumerate(dates):
        r_sum, b_sum = 0.0, 0.0
        for n in series:
            ds, ps = full[n]
            i = ds.index(d) if d in series[n] else None
            if i is None or i < WINDOW or i >= len(ps) - 1:
                continue
            ma = sum(ps[i - WINDOW + 1:i + 1]) / WINDOW
            rets = [ps[j] / ps[j - 1] - 1 for j in range(i - VOLW + 1, i + 1)]
            sd = statistics.pstdev(rets) * TD ** 0.5
            w = min(1.0, TARGET / sd) if (ps[i] > ma and sd > 0) else 0.0
            r = ps[i + 1] / ps[i] - 1
            r_sum += w * r + (1 - w) * (CASH / TD) - abs(w - prev_w[n]) * COST
            prev_w[n] = w
            b_sum += r
        port.append(r_sum / len(series))
        bah.append(b_sum / len(series))

    def perf(r):
        eq, peak, mdd = 1.0, 1.0, 0.0
        for v in r:
            eq *= 1 + v
            peak = max(peak, eq)
            mdd = max(mdd, 1 - eq / peak)
        sd = statistics.pstdev(r)
        return {"sharpe": (statistics.mean(r) / sd) * TD ** 0.5,
                "cagr": eq ** (TD / len(r)) - 1, "vol": sd * TD ** 0.5, "mdd": mdd}

    a, b = perf(port), perf(bah)
    print(f"\n{'方案':18s} {'夏普':>6s} {'年化':>8s} {'波动':>7s} {'回撤':>8s}")
    print(f"{'VOLT (日期对齐)':18s} {a['sharpe']:6.2f} {a['cagr']*100:7.2f}% "
          f"{a['vol']*100:6.2f}% {a['mdd']*100:7.1f}%")
    print(f"{'买入持有':18s} {b['sharpe']:6.2f} {b['cagr']*100:7.2f}% "
          f"{b['vol']*100:6.2f}% {b['mdd']*100:7.1f}%")
    print(f"\n结论：日期对齐后，VOLT 夏普 {a['sharpe']:.2f}（此前报告 1.88）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
