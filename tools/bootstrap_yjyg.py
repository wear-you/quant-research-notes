#!/usr/bin/env python3
"""bootstrap_yjyg.py — 用自助法量化「预增溢价」的抽样误差。

为什么需要：两次抽样（每季 25 只 vs 每季 40 只、且先按沪深300 分组）
给出 +2.04pp 和 +0.95pp / −1.23pp 三个不同结论。
如果季度间差异的置信区间宽到能同时容下这些数，说明**样本量根本不足以判定方向**。
"""
import json
import os
import random
import statistics

FIN = os.path.expanduser("~/finance")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def quarter_means(d, key):
    """d[key][q] = [exc...] → {q: mean}"""
    out = {}
    for q, v in d.get(key, {}).items():
        if len(v) >= 3:
            out[q] = statistics.mean(v)
    return out


def bootstrap_spread(a, b, n=5000, seed=7):
    """季度配对的自助法：对季度做重抽样。"""
    rng = random.Random(seed)
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    diffs = [a[q] - b[q] for q in common]
    means = []
    for _ in range(n):
        s = [rng.choice(diffs) for _ in diffs]
        means.append(statistics.mean(s))
    means.sort()
    return {"nq": len(common), "mean": statistics.mean(diffs),
            "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n)],
            "p_pos": sum(1 for x in means if x > 0) / n}


def main():
    res = load(os.path.join(FIN, "company", "resolve_yjyg.json"))
    ev = load(os.path.join(FIN, "company", "event_study.json"))

    print("=" * 70)
    print("A. 本次分组抽样（resolve_yjyg，每季每组最多 40 只）")
    print("=" * 70)
    g = {k: quarter_means(res, k) for k in res}
    for name, key in (("预增 HS300", "预增|HS300"), ("预增 NON300", "预增|NON300"),
                      ("预减 HS300", "预减|HS300"), ("预减 NON300", "预减|NON300")):
        v = g.get(key, {})
        if not v:
            continue
        print(f"{name:14s} 季度 {len(v):2d}  均值 {statistics.mean(v.values())*100:+6.2f}%")

    for label, (ka, kb) in (("NON300 内：预增−预减", ("预增|NON300", "预减|NON300")),
                            ("HS300 内：预增−预减", ("预增|HS300", "预减|HS300"))):
        if ka in g and kb in g:
            r = bootstrap_spread(g[ka], g[kb])
            if r:
                print(f"\n{label}")
                print(f"  季度数 {r['nq']}  价差 {r['mean']*100:+.2f}pp  "
                      f"95% 区间 [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]pp  "
                      f"P(>0) = {r['p_pos']*100:.0f}%")

    print("\n" + "=" * 70)
    print("B. 上一次抽样（event_study2，每季每组最多 25 只，未按指数分组）")
    print("=" * 70)
    rows = ev.get("rows", [])
    if rows:
        by = {}
        for r in rows:
            if r.get("adj5") is None:
                continue
            by.setdefault(r["type"], {}).setdefault(r["q"], []).append(r["adj5"])
        qm = {t: {q: statistics.mean(v) for q, v in d.items() if len(v) >= 3}
              for t, d in by.items()}
        for t, v in qm.items():
            if v:
                print(f"{t:6s} 季度 {len(v):2d}  均值 {statistics.mean(v.values())*100:+6.2f}%")
        if "预增" in qm and "预减" in qm:
            r = bootstrap_spread(qm["预增"], qm["预减"])
            if r:
                print(f"\n预增−预减（不分池）")
                print(f"  季度数 {r['nq']}  价差 {r['mean']*100:+.2f}pp  "
                      f"95% 区间 [{r['lo']*100:+.2f}, {r['hi']*100:+.2f}]pp  "
                      f"P(>0) = {r['p_pos']*100:.0f}%")

    print("\n" + "=" * 70)
    print("判读标准")
    print("=" * 70)
    print("  95% 区间跨过 0  → 方向未定，不能作为策略依据")
    print("  两次抽样区间不重叠 → 抽样规则本身在决定结论，更不能信")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
