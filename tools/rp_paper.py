#!/usr/bin/env python3
"""rp_paper.py — 风险平价组合的前向账本（append-only，不可回填）。

规则冻结于 2026-11-17：
  池子：data/ 下 33 个序列（剔除 csi_all / csi_div / nasdaq_etf）
  权重：1/vol60 归一化，单资产上限 35%
  缩放：组合目标波动 10%（相关性按 0.25 保守估计），不加杠杆
  过滤：价格低于 MA100 的资产不入选
  调仓：每 20 个交易日
  成本：单边 8bp，按权重变动计
  现金：未投资部分年化 1.5%

账本列：date, equity, invested, n_assets, turnover, weights(json)
"""
import csv
import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone

import market

FIN = os.path.expanduser("~/finance")
LEDGER = os.path.join(FIN, "rp_ledger.csv")
TZ = timezone(timedelta(hours=8))

TD = 244
TARGET = 0.10
CASH = 0.015
COST = 0.0008
REBAL = 20
VOLW = 60
MAW = 100
MAX_W = 0.35
MIN_BARS = 250
EXCLUDE = {"csi_all", "csi_div", "nasdaq_etf"}
COLS = ["date", "equity", "invested", "n_assets", "turnover", "weights"]


def load_all():
    out = []
    for f in sorted(os.listdir(os.path.join(FIN, "data"))):
        if f.endswith(".csv") and f[:-4] not in EXCLUDE:
            n = f[:-4]
            if len(market.load_series(n)) >= MIN_BARS:
                out.append(n)
    return out


def latest_state(names):
    px = {n: market.load_series(n) for n in names}
    dates = sorted(market.load_series("csi300"))
    d = dates[-1]
    w, vols = {}, {}
    for n in names:
        s = px[n]
        h = sorted(s)
        if d not in s or len(h) < max(VOLW, MAW) + 2:
            continue
        i = h.index(d)
        r = [s[h[j]] / s[h[j - 1]] - 1 for j in range(i - VOLW + 1, i + 1)]
        v = statistics.pstdev(r) * TD ** 0.5
        if v <= 0:
            continue
        ma = sum(s[h[j]] for j in range(i - MAW + 1, i + 1)) / MAW
        if s[d] <= ma:
            continue
        vols[n] = v
    if not vols:
        return d, {}, 0.0
    inv = {n: 1 / v for n, v in vols.items()}
    tot = sum(inv.values())
    w = {n: inv[n] / tot for n in inv}
    w = {n: min(v, MAX_W) for n, v in w.items()}
    s = sum(w.values())
    w = {n: v / s for n, v in w.items()}
    sig = math.sqrt(sum((w[n] * vols[n]) ** 2 for n in w)
                    + 0.25 * (sum(w[n] * vols[n] for n in w) ** 2
                              - sum((w[n] * vols[n]) ** 2 for n in w)))
    scale = min(1.0, TARGET / sig) if sig > 0 else 0.0
    return d, {n: v * scale for n, v in w.items()}, scale


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    return list(csv.DictReader(open(LEDGER)))


def main():
    names = load_all()
    d, w, scale = latest_state(names)
    rows = read_ledger()

    if not rows:
        rows.append({"date": d, "equity": "1.000000",
                     "invested": f"{sum(w.values()):.4f}",
                     "n_assets": str(len(w)), "turnover": "0",
                     "weights": json.dumps(w)})
        with open(LEDGER, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=COLS)
            wr.writeheader()
            wr.writerows(rows)
        print(f"初始化 {d}：{len(w)} 只，仓位 {sum(w.values())*100:.1f}%")
        return 0

    last = rows[-1]
    if last["date"] == d:
        print(f"{d} 已记录，跳过")
        return 0

    prev_w = json.loads(last["weights"])
    equity = float(last["equity"])

    # 用前一日的权重赚今天的收益
    ret = 0.0
    for n, wt in prev_w.items():
        s = market.load_series(n)
        h = sorted(s)
        if d in s and d in h and h.index(d) > 0:
            i = h.index(d)
            ret += wt * (s[d] / s[h[i - 1]] - 1)
    inv_prev = sum(prev_w.values())
    ret += (1 - inv_prev) * (CASH / TD)

    # 是否调仓（按交易日计数近似：每 20 条记录一次）
    trade = 0.0
    if len(rows) % REBAL == 0:
        trade = sum(abs(w.get(n, 0) - prev_w.get(n, 0)) for n in set(w) | set(prev_w))
        ret -= trade * COST

    equity *= (1 + ret)
    rows.append({"date": d, "equity": f"{equity:.6f}",
                 "invested": f"{sum(w.values()):.4f}",
                 "n_assets": str(len(w)), "turnover": f"{trade:.4f}",
                 "weights": json.dumps(w)})
    with open(LEDGER, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"{d} 净值 {equity:.4f}  当日 {ret*100:+.3f}%  "
          f"持仓 {len(w)} 只  仓位 {sum(w.values())*100:.0f}%"
          + (f"  调仓换手 {trade*100:.1f}%" if trade else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
