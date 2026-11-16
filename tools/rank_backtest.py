#!/usr/bin/env python3
"""rank_backtest.py — 横截面排序检验：不看单只股票涨跌，看"排序对不对"。

为什么换成这个提法：
  逐股方向预测 AUC 0.51（失败），但"预增组相对大盘 +2.04pp"（成立）。
  量化行业真正的做法是排序 + 分层，让大数定律替你工作。

核心指标：
  IC    = 每日横截面上「因子值」与「未来 5 日超额收益」的秩相关
  ICIR  = IC 均值 / IC 标准差（信息比）
  t     = ICIR × sqrt(有效天数)
  分层  = 顶 10% 与底 10% 的超额收益差

严格性：
  - 因子在 t 日收盘计算，收益取 t→t+5，无同 bar 执行
  - 超额 = 个股收益 − 沪深300 同期收益
  - 每日至少 30 只有效样本才计入
  - 分层换手率与成本单独报告

用法：python3 rank_backtest.py [持有天数 默认5]
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

FIN = os.path.expanduser("~/finance")
PX = os.path.join(FIN, "cache", "px.json")
ANN = os.path.join(FIN, "cache", "ann.json")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MIN_NAMES = 30
COST_BP = 15          # 双边合计，大市值现实区间 15–50bp
# 默认只用 2023 年起：公告缓存只覆盖该区间，且越往前"今天的成分股"幸存者偏差越重
START = "1900-01-01" if "--all" in sys.argv else "2023-01-01"

CAT_POS = ["业绩预增", "预增"]


def load(p, d):
    if not os.path.exists(p):
        return d
    try:
        return json.load(open(p))
    except Exception:
        print(f"⚠️  {os.path.basename(p)} 读取失败（可能正在被写入），按空处理")
        return d


def spearman(a, b):
    """秩相关，不依赖 scipy。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def bench():
    p = os.path.join(FIN, "data", "csi300.csv")
    return {r["date"]: float(r["close"]) for r in csv.DictReader(open(p))}


def build():
    px = load(PX, {})
    ann = load(ANN, {})
    bx = bench()
    bdates = sorted(bx)

    def bret(d0, d1):
        a, b = bx.get(d0), bx.get(d1)
        return (b / a - 1) if (a and b) else None

    rows = []
    for code, arr in px.items():
        if len(arr) < 150:
            continue
        dates = [x[0] for x in arr]
        close = np.array([x[1] for x in arr], dtype=float)
        vol = np.array([x[2] if len(x) > 2 else 0 for x in arr], dtype=float)
        rets = np.insert(np.diff(close) / close[:-1], 0, 0.0)
        byd = {}
        for d, t in ann.get(code, []):
            byd.setdefault(d, []).append(t)

        for k in range(65, len(arr) - H):
            d0, d1 = dates[k], dates[k + H]
            if d0 < START:
                continue
            br = bret(d0, d1)
            if br is None:
                continue
            fwd = close[k + H] / close[k] - 1
            vol20 = rets[k - 19:k + 1].std()
            e_pos = 0
            for j in range(max(0, k - 4), k + 1):
                for t in byd.get(dates[j], []):
                    if any(w in t for w in CAT_POS):
                        e_pos += 1
            rows.append({
                "date": d0, "code": code,
                "exc": fwd - br, "fwd": fwd,
                "ret5": close[k] / close[k - 5] - 1,
                "ret20": close[k] / close[k - 20] - 1,
                "ret60": close[k] / close[k - 60] - 1,
                "vol20": vol20,
                "ma20d": close[k] / close[k - 19:k + 1].mean() - 1,
                "vratio": (vol[k - 4:k + 1].mean() / vol[k - 19:k + 1].mean()
                           if vol[k - 19:k + 1].mean() > 0 else 1.0),
                "event_pos": e_pos,
            })
    return pd.DataFrame(rows)


FACTORS = [
    ("ret5", "-", "短期反转（A股强）"),
    ("ret20", "-", "20日反转"),
    ("ret60", "+", "中期动量（A股弱）"),
    ("vol20", "-", "低波动"),
    ("ma20d", "-", "偏离均线均值回归"),
    ("vratio", "-", "缩量"),
    ("event_pos", "+", "预增事件"),
]


def ic_series(df, col):
    out = []
    for d, g in df.groupby("date"):
        g = g.dropna(subset=[col, "exc"])
        if len(g) < MIN_NAMES:
            continue
        out.append((d, spearman(g[col].values, g["exc"].values)))
    return pd.Series(dict(out)).sort_index()


def decile_spread(df, col, sign):
    s = df.copy()
    s["score"] = s[col] * (1 if sign == "+" else -1)
    out = []
    for d, g in s.groupby("date"):
        g = g.dropna(subset=["score", "exc"])
        if len(g) < MIN_NAMES:
            continue
        try:
            q = pd.qcut(g["score"], 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        if q.nunique() < 10:
            continue
        top = g.loc[q == 9, "exc"].mean()
        bot = g.loc[q == 0, "exc"].mean()
        mtop = g.loc[q == 9, "exc"].median()
        mbot = g.loc[q == 0, "exc"].median()
        out.append((d, top, bot, top - bot, set(g.loc[q == 9, "code"]), mtop - mbot))
    return out


def stats(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 20:
        return None
    m, sd = x.mean(), x.std(ddof=1)
    return {"n": len(x), "mean": m, "std": sd,
            "ir": m / sd if sd else 0, "t": m / sd * np.sqrt(len(x)) if sd else 0}


def main():
    df = build()
    if len(df) < 5000:
        print(f"样本不足：{len(df)} 行。先跑 ml_direction.py 预热缓存。")
        return 1
    print(f"样本 {len(df):,} 行，{df['code'].nunique()} 只股票，"
          f"{df['date'].nunique()} 个交易日，{df['date'].min()} → {df['date'].max()}")
    print(f"持有 {H} 日，每日至少 {MIN_NAMES} 只\n")

    print("=" * 74)
    print("单因子 IC（秩相关：因子值 vs 未来超额收益）")
    print("=" * 74)
    print(f"{'因子':12s} {'方向':4s} {'IC均值':>8s} {'ICIR':>7s} {'t':>7s} {'天数':>6s}  说明")
    print("-" * 74)
    results = {}
    for col, sign, desc in FACTORS:
        ic = ic_series(df, col)
        if sign == "-":
            ic = -ic
        st = stats(ic.values)
        if not st:
            continue
        results[col] = (st, ic)
        flag = "✅" if abs(st["t"]) >= 3.0 else ("⚠️" if abs(st["t"]) >= 2.0 else "  ")
        print(f"{col:12s} {sign:4s} {st['mean']:+8.4f} {st['ir']:+7.3f} "
              f"{st['t']:+7.2f} {st['n']:6d}  {flag}{desc}")

    print("\n" + "=" * 74)
    print("分层：顶 10% − 底 10% 的超额收益")
    print("=" * 74)
    print(f"{'因子':12s} {'均值价差':>9s} {'t':>6s} {'中位价差':>9s} {'t':>6s} {'换手':>7s}")
    print("-" * 74)
    for col, sign, desc in FACTORS:
        sp = decile_spread(df, col, sign)
        if len(sp) < 20:
            continue
        st = stats([x[3] for x in sp])
        sm = stats([x[5] for x in sp])
        if not st or not sm:
            continue
        tos = []
        prev = None
        for x in sp:
            hold = x[4]
            if prev is not None and prev:
                tos.append(2 * (1 - len(hold & prev) / max(1, len(hold))))
            prev = hold
        to = np.mean(tos) if tos else 0.0
        print(f"{col:12s} {st['mean']*100:+8.3f}% {st['t']:+6.2f} "
              f"{sm['mean']*100:+8.3f}% {sm['t']:+6.2f} {to*100:6.1f}%")

    # 组合因子（方向按文献先验，事后不调）
    print("\n" + "=" * 74)
    print("组合因子：先用截面 z 分数合成，再看 IC")
    print("=" * 74)
    d = df.copy()
    for col, sign, _ in FACTORS:
        d[col] = d.groupby("date")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9))
        if sign == "-":
            d[col] = -d[col]
    d["combo"] = d[[c for c, _, _ in FACTORS]].mean(axis=1)
    ic = ic_series(d, "combo")
    st = stats(ic.values)
    if st:
        print(f"combo IC 均值 {st['mean']:+.4f}  ICIR {st['ir']:+.3f}  "
              f"t {st['t']:+.2f}  天数 {st['n']}")
    sp = decile_spread(d, "combo", "+")
    if len(sp) >= 20:
        diffs = [x[3] for x in sp]
        s2 = stats(diffs)
        print(f"分层价差日均 {s2['mean']*100:+.3f}%  t {s2['t']:+.2f}  "
              f"年化(未扣费) {s2['mean']*244/H*100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
