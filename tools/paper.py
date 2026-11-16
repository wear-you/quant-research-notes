#!/usr/bin/env python3
"""paper.py — 每日前向记录（paper trading，不涉及任何真实资金）。

每个资产跑三条独立账本：
  TREND : 收盘 > 200 日均线 → 满仓(1)，否则空仓(0)
  VOLT  : 趋势成立时，仓位 = min(1, 15% / 60日已实现年化波动)   ← 第二轮回测的最优变体
  BAH   : 始终持有（基准）

规则与 backtest2.py 完全一致：信号 t 日收盘产生，t+1 日按收盘价计入收益，
换仓扣非对称成本（A股ETF 买8bp/卖8bp；加密 买10bp/卖10bp）。
每天追加一行到 ledger.csv；equity 从 1.0 起算，**不回填历史**。

用法：python3 paper.py [--report]
"""
import csv
import glob
import os
import statistics
import sys
from datetime import datetime, timezone, timedelta

import market

DATA = os.path.expanduser("~/finance/data")
LEDGER = os.path.expanduser("~/finance/ledger.csv")
CN = (0.0008, 0.0008)
CR = (0.0010, 0.0010)
CRYPTO = {"btc", "eth", "sol", "bnb", "xrp", "ada", "doge", "link", "avax",
          "dot", "ltc", "bch", "trx"}
# 冻结参数：2016 年真样本外检验选出的配置（见 holdout.py），此后不再改动
WINDOW = 100
VOLW = 60
TARGET_VOL = 0.10
# 空仓部分的收益：按货币基金保守估计 1.5%/年（此前模型记为 0，低估了策略表现）
CASH_RATE = 0.015
TD = 244
TZ = timezone(timedelta(hours=8))


def asset_list():
    return sorted(os.path.basename(p)[:-4]
                  for p in glob.glob(os.path.join(DATA, "*.csv")))


def cost_for(name):
    return CR if name in CRYPTO else CN


def load(name):
    path = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(path):
        return []
    return [{"date": r["date"], "close": float(r["close"])}
            for r in csv.DictReader(open(path)) if float(r["close"]) > 0]


def trend_pos(prices):
    if len(prices) < WINDOW + 1:
        return 0
    return 1 if prices[-1] > sum(prices[-WINDOW:]) / WINDOW else 0


def volt_pos(prices):
    if len(prices) < max(WINDOW, VOLW) + 2:
        return 0.0
    if prices[-1] <= sum(prices[-WINDOW:]) / WINDOW:
        return 0.0
    rets = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - VOLW, len(prices))]
    sd = statistics.pstdev(rets) * TD ** 0.5
    if sd <= 0:
        return 0.0
    return round(min(1.0, TARGET_VOL / sd), 4)


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    return list(csv.DictReader(open(LEDGER)))


def last_state(rows, asset, strat):
    for r in reversed(rows):
        if r["asset"] == asset and r["strategy"] == strat:
            return float(r["equity"]), float(r["pos"]), r["date"], float(r["price"])
    return 1.0, 0.0, None, None


def portfolio_perf(all_rows, strat):
    """按日期对齐计算组合层绩效（使用 market.portfolio，禁止索引对齐）。"""
    per = {}
    for r in all_rows:
        if r["strategy"] != strat or not r.get("day_ret"):
            continue
        try:
            per.setdefault(r["asset"], {})[r["date"]] = float(r["day_ret"])
        except ValueError:
            continue
    if len(per) < 2:
        return None
    return market.perf(market.portfolio(per))


def main():
    want_report = "--report" in sys.argv
    rows = read_ledger()
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    out = []

    for name in asset_list():
        bars = load(name)
        if len(bars) < WINDOW + 2:
            continue
        prices = [b["close"] for b in bars]
        last_day, px = bars[-1]["date"], prices[-1]
        sig_t = float(trend_pos(prices))
        sig_v = volt_pos(prices)
        buy, sell = cost_for(name)
        cash_d = CASH_RATE / TD

        for strat, target in (("TREND", sig_t), ("VOLT", sig_v), ("BAH", 1.0)):
            eq, pos_prev, prev_day, prev_px = last_state(rows, name, strat)
            if prev_day is None or prev_day == last_day:
                eq_new, r = eq, 0.0
            else:
                r = px / prev_px - 1
                eq_new = eq * (1 + pos_prev * r + (1 - pos_prev) * cash_d)
                d = target - pos_prev
                if d > 0:
                    eq_new *= (1 - d * buy)
                elif d < 0:
                    eq_new *= (1 + d * sell)  # d 为负，等价于扣 (-d)*sell
            out.append({"date": last_day, "asset": name, "strategy": strat,
                        "price": f"{px:.4f}", "pos": f"{target:.4f}",
                        "equity": f"{eq_new:.6f}",
                        "day_ret": f"{r:.6f}" if prev_day and prev_day != last_day else "",
                        "logged_at": today})

    keep = [r for r in rows if out and r["date"] != out[0]["date"]] or []
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    fields = ["date", "asset", "strategy", "price", "pos", "equity", "day_ret", "logged_at"]
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep + out)

    print(f"{'资产':12s} {'止损':>6s} {'价位':>11s} {'TREND':>6s} {'VOLT':>6s} {'BAH净值':>9s}")
    seen = {}
    for r in out:
        seen.setdefault(r["asset"], {})[r["strategy"]] = r
    for name, s in seen.items():
        print(f"{name:12s} {'':6s} {float(s['TREND']['price']):11.2f} "
              f"{float(s['TREND']['pos']):6.1f} {float(s['VOLT']['pos']):6.2f} "
              f"{float(s['BAH']['equity']):9.4f}")

    if want_report and out:
        os.makedirs(os.path.expanduser("~/finance/reports"), exist_ok=True)
        path = os.path.expanduser(f"~/finance/reports/{out[0]['date']}.md")
        lines = [f"# 纸面跟踪 {out[0]['date']}", "",
                 "| 资产 | 收盘 | TREND | VOLT | TREND净值 | VOLT净值 | BAH净值 |",
                 "|---|---|---|---|---|---|---|"]
        for name, s in sorted(seen.items()):
            lines.append(
                f"| {name} | {float(s['TREND']['price']):.2f} | {float(s['TREND']['pos']):.1f} | "
                f"{float(s['VOLT']['pos']):.2f} | {float(s['TREND']['equity']):.4f} | "
                f"{float(s['VOLT']['equity']):.4f} | {float(s['BAH']['equity']):.4f} |")
        lines += ["", "## 组合层（日期对齐）", "",
                  "| 策略 | 夏普 | 年化 | 波动 | 回撤 | 样本 |", "|---|---|---|---|---|---|"]
        for st in ("VOLT", "TREND", "BAH"):
            p = portfolio_perf(rows + out, st)
            if p:
                lines.append(f"| {st} | {p['sharpe']:.2f} | {p['cagr']*100:.2f}% | "
                             f"{p['vol']*100:.2f}% | {p['mdd']*100:.1f}% | {p['n']} |")
            else:
                lines.append(f"| {st} | — | — | — | — | 积累中 |")
        lines += ["", "> 纸面记录，未涉及真实资金。参数冻结 MA100/10%，空仓计息 1.5%。",
                  "> 组合层只在共同交易日上计算（market.portfolio），禁止索引对齐。"]
        open(path, "w").write("\n".join(lines))
        print(f"\n报告: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
