#!/usr/bin/env python3
"""qforecast.py — 预测「收益分布」而不是「涨跌概率」，并用 pinball loss 严格验收。

为什么改：
  已验证 P(涨) 这条路技巧分数为负（5/5 资产）。
  但趋势信号影响的是收益**幅度**，所以改预测条件分位数：
  在(趋势×波动)状态下，未来 20 日收益的 10/25/50/75/90 分位是多少。

验收方式（分位数预测的标准指标）：
  pinball loss  L_q = max(q*(y-p), (q-1)*(y-p))
  覆盖率       实际值落在 q 分位以下的频率应 ≈ q
  对照基线     无条件历史分位数（"不管什么状态，都用整体分布"）
  技巧分数     1 - loss_model / loss_baseline，>0 才算有信息

严格因果：预测 t 时，只用 j+H <= t 的样本（结果已揭晓）。
"""
import csv
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
TD = 244
H = 20
WINDOW, VOLW = 100, 60
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
MIN_SAMPLES = 30


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
        recs.append({"i": i, "date": dates[i], "state": st,
                     "y": px[i + H] / px[i] - 1})
    return recs


def quantile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = q * (len(sorted_vals) - 1)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def pinball(y, p, q):
    d = y - p
    return max(q * d, (q - 1) * d)


def replay(recs):
    by_state, allv = {}, []
    ptr, out = 0, []
    for k, r in enumerate(recs):
        while ptr < k and recs[ptr]["i"] + H <= r["i"]:
            s = recs[ptr]["state"]
            by_state.setdefault(s, []).append(recs[ptr]["y"])
            allv.append(recs[ptr]["y"])
            ptr += 1
        st = by_state.get(r["state"], [])
        if len(st) >= MIN_SAMPLES:
            src, kind = sorted(st), "state"
        elif len(allv) >= MIN_SAMPLES:
            src, kind = sorted(allv), "base"
        else:
            continue
        out.append({**r, "kind": kind,
                    "pred": {q: quantile(src, q) for q in QS},
                    "base": {q: quantile(sorted(allv), q) for q in QS}})
    return out


def report(name, preds):
    if len(preds) < 200:
        print(f"{name:10s} 样本不足（{len(preds)}）")
        return None
    n = len(preds)
    loss_m = {q: sum(pinball(p["y"], p["pred"][q], q) for p in preds) / n for q in QS}
    loss_b = {q: sum(pinball(p["y"], p["base"][q], q) for p in preds) / n for q in QS}
    tm, tb = sum(loss_m.values()), sum(loss_b.values())
    skill = 1 - tm / tb if tb else 0

    print(f"\n{name}  样本 {n}  {preds[0]['date']} → {preds[-1]['date']}")
    print(f"  {'分位':>5s} {'pinball':>9s} {'基线':>9s} {'覆盖率':>8s} {'应为':>6s}")
    for q in QS:
        cov = sum(1 for p in preds if p["y"] <= p["pred"][q]) / n
        print(f"  {q:5.2f} {loss_m[q]:9.5f} {loss_b[q]:9.5f} {cov*100:7.1f}% {q*100:5.0f}%")
    print(f"  合计 pinball {tm:.5f} vs 基线 {tb:.5f}   技巧分数 {skill:+.4f}")

    med_err = [abs(p["y"] - p["pred"][0.5]) for p in preds]
    print(f"  中位预测 MAE {statistics.mean(med_err)*100:.2f}pp")
    return {"name": name, "n": n, "skill": skill, "loss": tm, "base": tb}


def main():
    names = sys.argv[1:] or ["moutai", "csi300", "sse_comp", "gold", "btc"]
    print("严格因果滚动 + pinball loss（分位数预测的标准指标）")
    res = []
    for nm in names:
        rows = load(nm)
        if len(rows) < WINDOW + VOLW + 300:
            print(f"{nm}: 数据不足")
            continue
        r = report(nm, replay(build(rows)))
        if r:
            res.append(r)
    if res:
        print("\n=== 汇总 ===")
        ok = [r for r in res if r["skill"] > 0]
        print(f"技巧分数为正：{len(ok)}/{len(res)}")
        for r in res:
            flag = "✅" if r["skill"] > 0.01 else ("⚠️" if r["skill"] > 0 else "❌")
            print(f"  {flag} {r['name']:10s} 技巧 {r['skill']:+.4f}")
        print("\n技巧分数 >0.01 才算有实质信息；接近 0 说明状态没带来新东西。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
