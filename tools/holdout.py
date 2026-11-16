#!/usr/bin/env python3
"""holdout.py — 真样本外检验（一次性，开封后不再调参）。

规则（写死在代码里，避免事后解释）：
  1. 分割日 2016-01-01。此前为样本内（IS），此后为样本外（OOS）。
  2. 只用 IS 数据挑参数（16 个配置）。
  3. 选定后冻结，一次性在 OOS 上评估。
  4. OOS 同时报告：冻结参数、先验参数（MA200/15%）、买入持有基准。

为什么是 2016：A 股此前经历 2005-2007 大牛、2008 崩盘、2010-2014 长熊、
2015 杠杆牛与股灾——覆盖了完整的极端周期。2016 之后（熔断、2018 熊、
2019-2021 结构牛、2022-2024 熊、2024-2026 反弹）对参数是完全未见过的。

不做的三件事：
  - 不在 OOS 上调参
  - 不在 OOS 上换资产池
  - 不为结果找解释
"""
import csv
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
TD = 244
CUT = "2016-01-01"
COST = 0.0008
MIN_BARS = 5000
WINDOWS = (100, 150, 200, 250)
TARGETS = (0.10, 0.15, 0.20, 0.30)
VOLW = 60


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]


def strat_returns(rows, window, target):
    """返回 [(date, daily_return)]，位置只用当日及以前的数据。"""
    px = [p for _, p in rows]
    dates = [d for d, _ in rows]
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, len(px))]
    out, prev = [], 0.0
    for i in range(max(window, VOLW), len(px) - 1):
        ma = sum(px[i - window + 1:i + 1]) / window
        if px[i] <= ma:
            w = 0.0
        else:
            sd = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
            w = min(1.0, target / sd) if sd > 0 else 0.0
        pnl = w * (px[i + 1] / px[i] - 1) - abs(w - prev) * COST
        out.append((dates[i], pnl))
        prev = w
    return out


def portfolio(per_asset, assets):
    """等权组合：每个资产先算自己的策略收益，再取平均。"""
    series = []
    for a in assets:
        s = per_asset[a]
        if s:
            series.append(s)
    L = min(len(s) for s in series)
    return [(series[0][i][0], sum(s[i][1] for s in series) / len(series))
            for i in range(L)]


def perf(series):
    r = [x for _, x in series]
    if len(r) < 30:
        return None
    eq, peak, mdd = 1.0, 1.0, 0.0
    for v in r:
        eq *= 1 + v
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    sd = statistics.pstdev(r)
    return {"n": len(r), "cagr": eq ** (TD / len(r)) - 1,
            "vol": sd * TD ** 0.5,
            "sharpe": (statistics.mean(r) / sd) * TD ** 0.5 if sd else 0.0,
            "mdd": mdd, "total": eq - 1}


def split(series):
    return ([x for x in series if x[0] < CUT], [x for x in series if x[0] >= CUT])


def main():
    names = sorted(os.path.basename(p)[:-4] for p in os.listdir(DATA) if p.endswith(".csv"))
    assets = [n for n in names if len(load(n)) >= MIN_BARS]
    print(f"资产（≥{MIN_BARS} 根）：{', '.join(assets)}\n")

    per = {}
    for a in assets:
        per[a] = {}

    configs = []
    for w in WINDOWS:
        for t in TARGETS:
            label = f"MA{w}/tgt{t:.0%}"
            full = {}
            for a in assets:
                full[a] = strat_returns(load(a), w, t)
            port = portfolio(full, assets)
            is_s, oos_s = split(port)
            configs.append((label, w, t, is_s, oos_s))

    print("=== 样本内（<2016）排名 ===")
    ranked = sorted(configs, key=lambda c: -(perf(c[3]) or {"sharpe": -9})["sharpe"])
    for lab, _, _, is_s, _ in ranked[:6]:
        p = perf(is_s)
        if p:
            print(f"  {lab:16s} 夏普 {p['sharpe']:5.2f}  年化 {p['cagr']*100:6.2f}%  "
                  f"回撤 {p['mdd']*100:5.1f}%  n={p['n']}")

    best = ranked[0]
    lab, w, t, is_s, oos_s = best
    print(f"\n冻结参数：{lab}（仅依据 2016 年前数据选出）\n")

    print("=== 样本外（≥2016）—— 只开这一次 ===")
    prior = next(c for c in configs if c[1] == 200 and abs(c[2] - 0.15) < 1e-9)

    rows_out = []
    for label, s in (("冻结参数 " + lab, oos_s),
                     ("先验参数 MA200/15%", prior[4])):
        p = perf(s)
        if p:
            rows_out.append((label, p))
            print(f"  {label:22s} 夏普 {p['sharpe']:5.2f}  年化 {p['cagr']*100:6.2f}%  "
                  f"波动 {p['vol']*100:5.2f}%  回撤 {p['mdd']*100:5.1f}%  累计 {p['total']*100:+7.1f}%")

    # 买入持有：等权
    bah = []
    for a in assets:
        rows = load(a)
        bah.append([(rows[i][0], rows[i + 1][1] / rows[i][1] - 1) for i in range(len(rows) - 1)])
    L = min(len(b) for b in bah)
    bah_port = [(bah[0][i][0], sum(b[i][1] for b in bah) / len(bah)) for i in range(L)]
    _, bah_oos = split(bah_port)
    p = perf(bah_oos)
    if p:
        rows_out.append(("买入持有（等权）", p))
        print(f"  {'买入持有（等权）':22s} 夏普 {p['sharpe']:5.2f}  年化 {p['cagr']*100:6.2f}%  "
              f"回撤 {p['mdd']*100:5.1f}%  累计 {p['total']*100:+7.1f}%")

    print("\n=== 判决 ===")
    frozen = perf(oos_s)
    base = p if p else {"sharpe": 0}
    if frozen and base["sharpe"]:
        d = frozen["sharpe"] - base["sharpe"]
        print(f"冻结参数 夏普 {frozen['sharpe']:.2f} vs 买入持有 {base['sharpe']:.2f}  差 {d:+.2f}")
        print("→ 跑赢基准" if d > 0 else "→ 跑输基准（策略在真样本外无效）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
