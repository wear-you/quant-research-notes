#!/usr/bin/env python3
"""market.py — 统一的数据对齐与组合计算层。

存在的唯一理由：**防止再次用索引当时间轴。**

规则（写死在这里，所有脚本共用）：
  1. 多资产组合只在**共同交易日**上计算，禁止按索引对齐
  2. 任何组合计算前先打印覆盖表
  3. 缺失日期不插值、不填充，直接剔除

用法：
  from market import load_series, panel, portfolio, perf, coverage

  names = ["csi300", "csi500", "sse50"]
  print_coverage(names)
  dates, px = panel(names)              # 共同交易日 + 对齐后的价格
"""
import csv
import math
import os
import statistics

DATA = os.path.expanduser("~/finance/data")
TD = 244


# ---------- 读取 ----------
def load_series(name, subdir=""):
    """返回 {date: close}，按日期排序由调用方处理。"""
    p = os.path.join(DATA, subdir, f"{name}.csv") if subdir else os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        try:
            v = float(r["close"]) if "close" in r else float(r["value"])
        except (KeyError, ValueError, TypeError):
            continue
        if v > 0:
            out[r["date"]] = v
    return out


# ---------- 覆盖表 ----------
def coverage_info(names):
    rows = []
    for n in names:
        s = load_series(n)
        if not s:
            rows.append({"name": n, "n": 0, "first": "-", "last": "-"})
            continue
        ds = sorted(s)
        rows.append({"name": n, "n": len(ds), "first": ds[0], "last": ds[-1]})
    return rows


def print_coverage(names, label=""):
    rows = coverage_info(names)
    print(f"=== 日期覆盖{'（' + label + '）' if label else ''} ===")
    print(f"{'资产':14s} {'根数':>6s} {'起':>12s} {'止':>12s}")
    for r in rows:
        print(f"{r['name']:14s} {r['n']:6d} {r['first']:>12s} {r['last']:>12s}")
    live = [r for r in rows if r["n"] > 0]
    if len(live) >= 2:
        first = max(r["first"] for r in live)
        last = min(r["last"] for r in live)
        common = set.intersection(*[set(load_series(r["name"])) for r in live])
        print(f"共同区间 {first} → {last}，共同交易日 {len(common)} 天")
        print(f"注意：若共同交易日远少于最短序列，说明资产历史差异大，任何按索引对齐都是错的。")
    return rows


# ---------- 对齐 ----------
def panel(names):
    """返回 (dates, {name: [price...]})，只保留所有资产都有数据的交易日。"""
    series = {n: load_series(n) for n in names}
    series = {n: s for n, s in series.items() if s}
    if len(series) < 2:
        raise ValueError("至少需要 2 个有数据的资产")
    common = sorted(set.intersection(*[set(s) for s in series.values()]))
    px = {n: [s[d] for d in common] for n, s in series.items()}
    return common, px


def returns(px):
    out = [0.0]
    for i in range(1, len(px)):
        out.append(px[i] / px[i - 1] - 1)
    return out


# ---------- 组合 ----------
def portfolio(per_asset):
    """per_asset: {name: {date: return}} → [(date, avg_return)]，只保留全部资产都有的日期。"""
    if not per_asset:
        return []
    common = set.intersection(*[set(v) for v in per_asset.values()])
    dates = sorted(common)
    k = len(per_asset)
    return [(d, sum(v[d] for v in per_asset.values()) / k) for d in dates]


# ---------- 绩效 ----------
def perf(daily, td=TD):
    """daily: list of returns（或 [(date, ret)]）。"""
    r = [x[1] if isinstance(x, tuple) else x for x in daily]
    if len(r) < 30:
        return None
    eq, peak, mdd = 1.0, 1.0, 0.0
    for v in r:
        eq *= 1 + v
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    sd = statistics.pstdev(r)
    return {"n": len(r),
            "cagr": eq ** (td / len(r)) - 1,
            "vol": sd * td ** 0.5,
            "sharpe": (statistics.mean(r) / sd) * td ** 0.5 if sd else 0.0,
            "mdd": mdd, "total": eq - 1}


def print_perf(label, p):
    if not p:
        print(f"{label:18s} 样本不足")
        return
    print(f"{label:18s} 夏普 {p['sharpe']:5.2f}  年化 {p['cagr']*100:6.2f}%  "
          f"波动 {p['vol']*100:6.2f}%  回撤 {p['mdd']*100:6.1f}%  n={p['n']}")


# ---------- 守卫 ----------
def assert_same_dates(a, b, tag=""):
    """在把两个序列相加/平均之前调用，防止隐式错位。"""
    sa, sb = set(a), set(b)
    if sa != sb:
        raise AssertionError(
            f"{tag} 日期不一致：仅A {len(sa-sb)} 天，仅B {len(sb-sa)} 天。"
            "请先用 panel()/portfolio() 对齐，不要按索引拼接。")
    return True


if __name__ == "__main__":
    import glob
    names = sorted(os.path.basename(p)[:-4]
                   for p in glob.glob(os.path.join(DATA, "*.csv")))
    print_coverage(names, "全部资产")
    print("\n提示：共同交易日越少，说明资产历史差异越大——这种池子绝不能按索引对齐。")
