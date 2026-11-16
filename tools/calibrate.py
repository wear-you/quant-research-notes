#!/usr/bin/env python3
"""calibrate.py — 用历史数据检验「预测协议本身」有没有信息量。

问题：等 20 个交易日太慢，而且 2 条样本什么都说明不了。
做法：把 predict.py 的方法在历史上**滚动重放**——每个时点只用当时已知的信息
（严格因果：只允许使用"结果已揭晓"的历史样本），然后统计 Brier 与校准。

这不能替代真实前向验证（同一批数据仍被反复使用），
但它能立刻回答一个致命问题：**这套方法有没有可能打败 0.25（无信息线）？**

判据：
  Brier < 0.25           → 有信息
  Brier < 基准 Brier     → 比"永远猜基准概率"更好（skill score > 0）
  校准表接近 45° 线       → 概率可信
"""
import csv
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
TD = 244
H = 20
WINDOW, VOLW = 100, 60
MIN_SAMPLES = 20


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]


def build(rows):
    px = [p for _, p in rows]
    dates = [d for d, _ in rows]
    n = len(px)
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, n)]
    vol = [0.0] * n
    for i in range(VOLW, n):
        vol[i] = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
    v = [x for x in vol[VOLW:] if x > 0]
    med = statistics.median(v) if v else 0

    recs = []
    for i in range(WINDOW, n - H):
        ma = sum(px[i - WINDOW + 1:i + 1]) / WINDOW
        st = ("up" if px[i] > ma else "down") + ("hi" if vol[i] > med else "lo")
        fwd = px[i + H] / px[i] - 1
        recs.append({"i": i, "date": dates[i], "state": st,
                     "up": 1 if fwd > 0 else 0, "ret": fwd})
    return recs, px, dates


def replay(recs):
    """严格因果滚动：预测 t 时，只能用 j+H <= t 的样本。"""
    by_state, allc = {}, [0, 0]          # [ups, total]
    out, ptr = [], 0
    for k, r in enumerate(recs):
        while ptr < k and recs[ptr]["i"] + H <= r["i"]:
            s = recs[ptr]["state"]
            c = by_state.setdefault(s, [0, 0])
            c[0] += recs[ptr]["up"]; c[1] += 1
            allc[0] += recs[ptr]["up"]; allc[1] += 1
            ptr += 1
        c = by_state.get(r["state"], [0, 0])
        if c[1] >= MIN_SAMPLES:
            p = (c[0] + 0.5) / (c[1] + 1)
            kind = "state"
        elif allc[1] >= MIN_SAMPLES:
            p = (allc[0] + 0.5) / (allc[1] + 1)
            kind = "base"
        else:
            continue
        out.append({**r, "p": p, "kind": kind,
                    "p_base": (allc[0] + 0.5) / (allc[1] + 1) if allc[1] else 0.5})
    return out


def report(name, preds):
    if len(preds) < 100:
        print(f"{name:10s} 样本不足（{len(preds)}）")
        return None
    n = len(preds)
    brier = sum((p["p"] - p["up"]) ** 2 for p in preds) / n
    brier_base = sum((p["p_base"] - p["up"]) ** 2 for p in preds) / n
    brier_coin = 0.25
    hit = sum(1 for p in preds if (p["p"] > 0.5) == (p["up"] == 1)) / n
    skill = 1 - brier / brier_base if brier_base else 0
    print(f"\n{name}  样本 {n}  区间 {preds[0]['date']} → {preds[-1]['date']}")
    print(f"  Brier {brier:.4f}   基准 {brier_base:.4f}   硬币 {brier_coin:.4f}")
    print(f"  技巧分数 {skill:+.3f}（>0 表示比『永远猜基准』好）")
    print(f"  方向命中率 {hit*100:.1f}%")
    print(f"  {'预测区间':12s} {'条数':>5s} {'预测均值':>8s} {'实际上涨':>8s} {'偏差':>7s}")
    for lo, hi in [(0, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1.01)]:
        sel = [p for p in preds if lo <= p["p"] < hi]
        if len(sel) < 20:
            continue
        pm = sum(p["p"] for p in sel) / len(sel)
        ar = sum(p["up"] for p in sel) / len(sel)
        print(f"  {lo:.2f}–{hi:.2f}      {len(sel):5d} {pm*100:7.1f}% {ar*100:7.1f}% "
              f"{(ar-pm)*100:+6.1f}pp")
    return {"name": name, "n": n, "brier": brier, "brier_base": brier_base, "skill": skill}


def main():
    names = sys.argv[1:] or ["moutai", "csi300", "sse_comp", "gold", "btc"]
    print("严格因果滚动重放（预测 t 时只用结果已揭晓的样本）")
    res = []
    for nm in names:
        rows = load(nm)
        if len(rows) < WINDOW + VOLW + 200:
            print(f"{nm}: 数据不足")
            continue
        recs, _, _ = build(rows)
        preds = replay(recs)
        r = report(nm, preds)
        if r:
            res.append(r)

    if res:
        print("\n=== 汇总 ===")
        ok = [r for r in res if r["skill"] > 0]
        print(f"技巧分数为正：{len(ok)}/{len(res)}")
        for r in res:
            flag = "✅" if r["skill"] > 0.02 else ("⚠️" if r["skill"] > 0 else "❌")
            print(f"  {flag} {r['name']:10s} Brier {r['brier']:.4f}  技巧 {r['skill']:+.3f}")
        print("\n注意：这是同一批数据上的滚动检验，不等于真实前向验证。"
              "\n技巧分数接近 0 说明该状态几乎没有信息量。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
