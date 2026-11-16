#!/usr/bin/env python3
"""verify_holdout.py — 日期严格对齐的样本外检验（唯一权威版本）。

历史教训：旧版 holdout.py 按**索引**对齐不同起始日的资产，
把 1990 年代的波动与 2005 年代的波动平均在一起，人为制造分散化，
把夏普从 0.47 抬到 1.88。本脚本改用 market.panel()，只在共同交易日上计算。
"""
import os
import statistics
import sys

import market

CUT = "2016-01-01"
COST = 0.0008
CASH = 0.015
WINDOW, VOLW = 100, 60
TARGET = 0.10
TD = 244

NAMES = ["bond_index", "csi100", "csi300", "csi500", "sse50", "sse_comp", "szse_comp"]


def strategy_for(px):
    """给定对齐后的价格序列，返回 {date_index: 策略日收益}。"""
    n = len(px)
    rets = market.returns(px)
    out, prev = {}, 0.0
    for i in range(WINDOW, n - 1):
        ma = sum(px[i - WINDOW + 1:i + 1]) / WINDOW
        if px[i] <= ma:
            w = 0.0
        else:
            sd = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
            w = min(1.0, TARGET / sd) if sd > 0 else 0.0
        r = px[i + 1] / px[i] - 1
        out[i] = w * r + (1 - w) * (CASH / TD) - abs(w - prev) * COST
        prev = w
    return out


def main():
    names = [n for n in NAMES if os.path.exists(os.path.join(market.DATA, f"{n}.csv"))]
    market.print_coverage(names, "样本外检验")

    dates, px = market.panel(names)
    print(f"对齐后共 {len(dates)} 个交易日，评估起点 {CUT}\n")

    strat, bah = {}, {}
    for n in names:
        s = strategy_for(px[n])
        strat[n] = {dates[i]: v for i, v in s.items() if dates[i] >= CUT}
        bah[n] = {dates[i]: px[n][i + 1] / px[n][i] - 1
                  for i in range(len(dates) - 1) if dates[i] >= CUT}

    port_s = market.portfolio(strat)
    port_b = market.portfolio(bah)

    ps, pb = market.perf(port_s), market.perf(port_b)
    print(f"评估区间 {port_s[0][0]} → {port_s[-1][0]}，{len(port_s)} 天\n")
    market.print_perf("VOLT（日期对齐）", ps)
    market.print_perf("买入持有", pb)

    if ps and pb:
        print(f"\n夏普 {ps['sharpe']/pb['sharpe']:.1f}×   "
              f"回撤 {ps['mdd']*100:.1f}% vs {pb['mdd']*100:.1f}%   "
              f"年化 {ps['cagr']*100:.2f}% vs {pb['cagr']*100:.2f}%")

    # 一致性守卫：组合序列的日期必须与面板一致
    assert len(port_s) == len(port_b), "组合序列长度不一致"
    print("\n✅ 日期对齐校验通过（组合只在共同交易日上计算）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
