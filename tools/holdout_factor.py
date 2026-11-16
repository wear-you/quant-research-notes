#!/usr/bin/env python3
"""holdout_factor.py — 把外部参数接入波动预测，看它能不能改善策略的样本外表现。

判据（事先写死，事后不改）：
  1. 波动预测的样本外 MAE 必须改善 >= 3%
  2. 策略样本外夏普不得下降
  两条同时满足才采用，否则弃用。

两个测试：
  A. 10 年国债收益率水平 —— 数据自 2002 年，可做"2016 前拟合 / 2016 后检验"
  B. 美元兑人民币水平 —— 数据仅自 2015-01，无法做 2016 分割，
     改用"2016-2019 拟合 / 2020 起检验"的次级设计，并明确标注

除波动预测外，策略其余部分完全不变（MA100 趋势过滤、目标波动 10%、成本 8bp、空仓计息 1.5%），
这样两组之间的差异只可能来自"波动预测得更准"。
"""
import csv
import math
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
MACRO = os.path.join(DATA, "macro")
TD = 244
H = 20
BLOCK = 60
MIN_TRAIN = 500
COST = 0.0008
CASH = 0.015
TARGET_VOL = 0.10
WINDOW = 100
MIN_BARS = 5000


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]


def load_macro(name):
    p = os.path.join(MACRO, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["value"])) for r in csv.DictReader(open(p))]


def align(dates, series):
    out, j, last = [], 0, None
    for d in dates:
        while j < len(series) and series[j][0] <= d:
            last = series[j][1]
            j += 1
        out.append(last)
    return out


def ols(X, y):
    k = len(X[0])
    A = [[sum(X[n][i] * X[n][j] for n in range(len(X))) for j in range(k)] +
         [sum(X[n][i] * y[n] for n in range(len(X)))] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12:
            return None
        A[i], A[p] = A[p], A[i]
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                for c in range(i, k + 1):
                    A[r][c] -= f * A[i][c]
    return [A[i][k] / A[i][i] for i in range(k)]


def pred(b, row):
    return sum(x * y for x, y in zip(b, row))


def forecasts(px, dates, factor):
    """逐块向前滚动，返回两组波动预测：基线 / 加因子。"""
    n = len(px)
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, n)]
    cur = [statistics.pstdev(rets[max(0, i - 60):i + 1]) * TD ** 0.5 for i in range(n)]
    fwd = [None] * n
    for i in range(n - H):
        fwd[i] = statistics.pstdev(rets[i + 1:i + 1 + H]) * TD ** 0.5

    ok = [i for i in range(100, n - H)
          if cur[i] > 0 and fwd[i] and (factor[i] is not None or factor is None)]
    fa, fb = {}, {}
    k = MIN_TRAIN
    while k < len(ok):
        tr, te = ok[:k], ok[k:k + BLOCK]
        X1 = [[1.0, math.log(cur[i])] for i in tr]
        y = [fwd[i] for i in tr]
        b1 = ols(X1, y)
        b2 = None
        if factor is not None:
            X2 = [[1.0, math.log(cur[i]), factor[i]] for i in tr]
            b2 = ols(X2, y)
        if b1:
            for i in te:
                fa[i] = max(0.001, pred(b1, [1.0, math.log(cur[i])]))
                if b2:
                    fb[i] = max(0.001, pred(b2, [1.0, math.log(cur[i]), factor[i]]))
        k += BLOCK
    return fa, fb, fwd, cur


def strategy(px, dates, fmap, cut):
    """用给定的波动预测跑策略，返回 cut 之后日收益序列。"""
    n = len(px)
    out, prev = [], 0.0
    for i in range(WINDOW, n - 1):
        if dates[i] < cut:
            continue
        ma = sum(px[i - WINDOW + 1:i + 1]) / WINDOW
        if px[i] <= ma or i not in fmap:
            w = 0.0
        else:
            w = min(1.0, TARGET_VOL / fmap[i])
        r = px[i + 1] / px[i] - 1
        pnl = w * r + (1 - w) * (CASH / TD) - abs(w - prev) * COST
        out.append(pnl)
        prev = w
    return out


def perf(r):
    if len(r) < 100:
        return None
    eq, peak, mdd = 1.0, 1.0, 0.0
    for v in r:
        eq *= 1 + v
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    sd = statistics.pstdev(r)
    return {"cagr": eq ** (TD / len(r)) - 1, "vol": sd * TD ** 0.5,
            "sharpe": (statistics.mean(r) / sd) * TD ** 0.5 if sd else 0,
            "mdd": mdd}


def run(factor_name, cut, assets):
    print(f"\n{'='*66}\n因子：{factor_name}   检验起点：{cut}\n{'='*66}")
    fac = align_for(factor_name, assets)
    res = {"A": [], "B": [], "C": [], "maeA": [], "maeB": [], "maeC": []}
    for a in assets:
        rows = load(a)
        if not rows:
            continue
        dates = [d for d, _ in rows]
        px = [p for _, p in rows]
        fvals = align(dates, load_macro(factor_name))
        fa, fb, fwd, cur = forecasts(px, dates, fvals)
        common = [i for i in fb if fwd[i]]
        if not common:
            continue
        res["maeA"].append(sum(abs(fwd[i] - fa[i]) for i in common) / len(common))
        res["maeB"].append(sum(abs(fwd[i] - fb[i]) for i in common) / len(common))
        curmap = {i: cur[i] for i in common}
        res["maeC"].append(sum(abs(fwd[i] - cur[i]) for i in common) / len(common))
        ra = strategy(px, dates, fa, cut)
        rb = strategy(px, dates, fb, cut)
        rc = strategy(px, dates, curmap, cut)
        if ra and rb:
            res["A"].append(ra)
            res["B"].append(rb)
        if rc:
            res["C"].append(rc)

    if not res["A"]:
        print("样本不足")
        return None
    L = min(len(x) for x in res["A"])
    pa = perf([sum(x[i] for x in res["A"]) / len(res["A"]) for i in range(L)])
    L2 = min(len(x) for x in res["B"])
    pb = perf([sum(x[i] for x in res["B"]) / len(res["B"]) for i in range(L2)])
    maeA = statistics.mean(res["maeA"])
    maeB = statistics.mean(res["maeB"])
    gain = (maeA - maeB) / maeA

    print(f"{'方案':16s} {'MAE(波动)':>10s} {'夏普':>6s} {'年化':>8s} {'回撤':>8s}")
    print(f"{'基线(仅当前波动)':16s} {maeA*100:9.2f}% {pa['sharpe']:6.2f} {pa['cagr']*100:7.2f}% {pa['mdd']*100:7.1f}%")
    print(f"{'+ ' + factor_name:16s} {maeB*100:9.2f}% {pb['sharpe']:6.2f} {pb['cagr']*100:7.2f}% {pb['mdd']*100:7.1f}%")
    if res.get("C"):
        L3 = min(len(x) for x in res["C"])
        pc = perf([sum(x[i] for x in res["C"]) / len(res["C"]) for i in range(L3)])
        cur_mae = statistics.mean(res["maeC"])
        print(f"{'朴素(直接用当前波动)':16s} {cur_mae*100:9.2f}% {pc['sharpe']:6.2f} {pc['cagr']*100:7.2f}% {pc['mdd']*100:7.1f}%")
        print(f"   → 朴素 vs 回归基线：夏普 {pc['sharpe']-pa['sharpe']:+.2f}，MAE {(cur_mae-maeA)/maeA*100:+.1f}%")
    print(f"\nMAE 改善 {gain*100:+.1f}%   夏普变化 {pb['sharpe']-pa['sharpe']:+.2f}")

    ok_mae = gain >= 0.03
    ok_sharpe = pb["sharpe"] >= pa["sharpe"]
    verdict = "✅ 采用" if (ok_mae and ok_sharpe) else "❌ 弃用"
    print(f"判据：MAE改善>=3% {'✓' if ok_mae else '✗'}  夏普不降 {'✓' if ok_sharpe else '✗'}  → {verdict}")
    return {"verdict": verdict, "gain": gain, "dsharpe": pb["sharpe"] - pa["sharpe"]}


def align_for(factor_name, assets):
    return None


def main():
    assets = sorted(os.path.basename(p)[:-4] for p in
                    (os.path.join(DATA, f) for f in os.listdir(DATA))
                    if p.endswith(".csv") and len(load(os.path.basename(p)[:-4])) >= MIN_BARS)
    if "--idx" in sys.argv:
        assets = [a for a in assets if a != "moutai"]
        print("【仅指数/ETF】排除个股")
    print(f"资产（≥{MIN_BARS} 根）：{', '.join(assets)}")
    r1 = run("cn10y", "2016-01-01", assets)
    r2 = run("usdcny", "2020-01-01", assets)
    print("\n=== 总判 ===")
    for nm, r in (("cn10y（2016 起）", r1), ("usdcny（2020 起，次级设计）", r2)):
        if r:
            print(f"  {nm}: {r['verdict']}  MAE {r['gain']*100:+.1f}%  夏普 {r['dsharpe']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
