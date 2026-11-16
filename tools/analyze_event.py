#!/usr/bin/env python3
"""analyze_event.py — 对 event_study.json 做簇稳健检验。

为什么要簇稳健：484 条观测大量共享同一时间窗口，彼此不独立，
直接用标准误算 t 值会严重高估显著性。正确做法是把同一季度的观测
先聚成一个点（季度均值），再对季度做 t 检验——这样只有 13 个独立观测。
"""
import json
import os
import statistics

P = os.path.expanduser("~/finance/company/event_study.json")
d = json.load(open(P))
rows = d["rows"]


def cluster(metric, gtype):
    by = {}
    for r in rows:
        if r["type"] != gtype or r.get(metric) is None:
            continue
        by.setdefault(r["q"], []).append(r[metric])
    return {q: statistics.mean(v) for q, v in by.items() if len(v) >= 3}


def tstat(v):
    if len(v) < 3:
        return 0.0, 0.0
    m = statistics.mean(v)
    se = statistics.stdev(v) / len(v) ** 0.5
    return m, (m / se if se else 0.0)


print("=" * 60)
print("簇稳健检验（每季度聚成一个观测）")
print("=" * 60)
for metric, label in (("adj5", "超额+5日"), ("adj20", "超额+20日")):
    a = cluster(metric, "预增")
    b = cluster(metric, "预减")
    common = sorted(set(a) & set(b))
    spreads = [a[q] - b[q] for q in common]
    m, t = tstat(spreads)
    print(f"\n{label}：共同季度 {len(common)} 个")
    print(f"  预增均值 {statistics.mean(a.values())*100:+.2f}%   "
          f"预减均值 {statistics.mean(b.values())*100:+.2f}%")
    print(f"  价差均值 {m*100:+.2f}pp   t = {t:.2f}   "
          f"正的季度 {sum(1 for s in spreads if s>0)}/{len(spreads)}")
    print(f"  逐季价差: " + ", ".join(f"{q[2:6]}:{s*100:+.1f}" for q, s in zip(common, spreads)))

# 只用预增的绝对超额（含费）
a5 = cluster("adj5", "预增")
cost = 0.0025  # 双边：佣金+印花税+滑点，A股 5 日持有一次进出
net = {q: v - cost for q, v in a5.items()}
m, t = tstat(list(net.values()))
print(f"\n预增组扣 25bp 成本后：季度均值 {m*100:+.2f}%，t = {t:.2f}")

# 去掉最好的一个季度，看是否由单季撑起
if len(a5) > 4:
    worst_removed = sorted(a5.items(), key=lambda x: -x[1])[1:]
    m2, t2 = tstat([v for _, v in worst_removed])
    print(f"剔除最强季度后：季度均值 {m2*100:+.2f}%，t = {t2:.2f}")

# 报价差稳定性
full = cluster("adj5", "预增")
print(f"\n预增超额+5日在 {len(full)} 个季度中为正："
      f"{sum(1 for v in full.values() if v>0)}/{len(full)}")
