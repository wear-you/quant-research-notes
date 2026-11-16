#!/usr/bin/env python3
"""reflect.py — 结算 + 反思闭环（对应 TradingAgents 的 settlement.py + reflection.py）。

思路：
  1. 结算（settlement）：历史每一个时点，信号说"持仓"还是"空仓"，
     之后 20 个交易日的真实结果是什么。信号不是结论，结算后的统计才是。
  2. 反思（reflection）：把结算结果按状态分组——趋势向上/向下、波动高/低、资产——
     找出"信号在什么状态下失效"。

这是纸面账本缺的那一半。账本只记"我今天做了什么"，
结算回答"我做的对不对"，反思回答"什么时候我会错"。

用法：
  python3 reflect.py            # 全历史结算 + 反思报告
  python3 reflect.py --ledger   # 只用 paper.py 的账本
"""
import csv
import glob
import os
import statistics
import sys
from datetime import datetime, timezone, timedelta

DATA = os.path.expanduser("~/finance/data")
LEDGER = os.path.expanduser("~/finance/ledger.csv")
OUTDIR = os.path.expanduser("~/finance/reflection")
TD = 244
WINDOW, VOLW, HORIZON = 200, 60, 20
TZ = timezone(timedelta(hours=8))


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]


def settle_asset(name, rows):
    """对每个历史时点，记录信号状态与随后 HORIZON 日收益。"""
    px = [p for _, p in rows]
    n = len(px)
    if n < WINDOW + HORIZON + 5:
        return []
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, n)]
    out = []
    for i in range(WINDOW, n - HORIZON):
        ma = sum(px[i - WINDOW + 1:i + 1]) / WINDOW
        up = px[i] > ma
        sd = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
        fwd = px[i + HORIZON] / px[i] - 1
        out.append({"asset": name, "date": rows[i][0], "up": up,
                    "vol": sd, "fwd": fwd,
                    "trend_pos": 1.0 if up else 0.0,
                    "volt_pos": min(1.0, 0.15 / sd) if (up and sd > 0) else 0.0})
    return out


def agg(rows, key_filter):
    sel = [r for r in rows if key_filter(r)]
    if len(sel) < 30:
        return None
    fwd = [r["fwd"] for r in sel]
    return {"n": len(sel), "mean": statistics.mean(fwd),
            "median": statistics.median(fwd),
            "win": sum(1 for f in fwd if f > 0) / len(fwd),
            "sd": statistics.pstdev(fwd) if len(fwd) > 1 else 0}


def fmt(d):
    if not d:
        return "样本不足"
    return (f"n={d['n']:5d}  均值 {d['mean']*100:+6.2f}%  中位 {d['median']*100:+6.2f}%  "
            f"胜率 {d['win']*100:5.1f}%  波动 {d['sd']*100:5.2f}%")


def main():
    use_ledger = "--ledger" in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    today = datetime.now(TZ).strftime("%Y-%m-%d")

    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.csv")))
    all_rows = []
    for n in names:
        rows = load(n)
        if rows:
            all_rows.extend(settle_asset(n, rows))

    L = []
    L.append(f"# 结算与反思 · {today}")
    L.append("")
    L.append(f"> 全历史决策结算（{len(names)} 个资产，前瞻 {HORIZON} 个交易日）。")
    L.append("> 回答两个问题：信号有没有区分度？在什么状态下失效？")
    L.append("")

    if not all_rows:
        L.append("没有足够的结算样本。")
        open(os.path.join(OUTDIR, f"{today}.md"), "w").write("\n".join(L))
        print("\n".join(L))
        return 0

    L.append("## 一、信号区分度（核心问题）")
    L.append("")
    L.append("如果信号有效，**持仓时的未来收益应显著高于空仓时**。")
    L.append("")
    L.append("| 状态 | TREND 信号 | 统计 |")
    L.append("|---|---|---|")
    L.append(f"| 趋势向上（持仓） | 1 | {fmt(agg(all_rows, lambda r: r['up']))} |")
    L.append(f"| 趋势向下（空仓） | 0 | {fmt(agg(all_rows, lambda r: not r['up']))} |")
    L.append("")

    up = agg(all_rows, lambda r: r["up"])
    dn = agg(all_rows, lambda r: not r["up"])
    spread = (up["mean"] - dn["mean"]) if (up and dn) else None

    L.append("## 二、判决")
    L.append("")
    if spread is None:
        L.append("样本不足，无法判决。")
    else:
        L.append(f"**持仓期前瞻收益 − 空仓期前瞻收益 = {spread*100:+.2f} 个百分点**")
        L.append("")
        if spread > 0.01:
            L.append("→ 信号有正向区分度：趋势向上时未来 20 日确实更好。")
        elif spread > 0:
            L.append("→ 区分度很弱（<1pp），考虑成本后大概率归零。")
        else:
            L.append("→ **信号方向是反的**：趋势向下时未来收益反而更高（均值回归占主导）。")
    L.append("")

    L.append("## 三、分状态拆解：什么时候失效")
    L.append("")
    L.append("| 状态 | 统计 |")
    L.append("|---|---|")
    vol_hi = statistics.median([r["vol"] for r in all_rows]) if all_rows else 0
    segs = [
        ("趋势↑ 且 低波动", lambda r: r["up"] and r["vol"] <= vol_hi),
        ("趋势↑ 且 高波动", lambda r: r["up"] and r["vol"] > vol_hi),
        ("趋势↓ 且 低波动", lambda r: not r["up"] and r["vol"] <= vol_hi),
        ("趋势↓ 且 高波动", lambda r: not r["up"] and r["vol"] > vol_hi),
    ]
    for label, f in segs:
        L.append(f"| {label} | {fmt(agg(all_rows, f))} |")
    L.append("")

    L.append("## 四、分资产：哪些标的根本不适合这套信号")
    L.append("")
    L.append("| 资产 | 持仓期均值 | 空仓期均值 | 差值 |")
    L.append("|---|---|---|---|")
    per = {}
    for r in all_rows:
        per.setdefault(r["asset"], []).append(r)
    bad = []
    for a, rs in sorted(per.items()):
        u = agg(rs, lambda r: r["up"])
        d = agg(rs, lambda r: not r["up"])
        if u and d:
            diff = u["mean"] - d["mean"]
            L.append(f"| {a} | {u['mean']*100:+.2f}% | {d['mean']*100:+.2f}% | {diff*100:+.2f}pp |")
            if diff < 0:
                bad.append(a)
    L.append("")
    if bad:
        L.append(f"**信号方向为负的资产（{len(bad)} 个）**：{', '.join(bad)}")
        L.append("")
        L.append("这些标的上，趋势跟随是反着做的——不是参数问题，是标的特性（均值回归型）。")
        L.append("")

    L.append("## 五、对纸面账本的含义")
    L.append("")
    L.append("- 账本记录的是「我做了什么」，本文件回答「做得对不对」。")
    L.append("- 如果某一类状态长期为负，应该在那类状态下降仓，而不是调参数。")
    L.append(f"- 下次复跑：`python3 reflect.py`，报告落在 `~/finance/reflection/{today}.md`。")

    path = os.path.join(OUTDIR, f"{today}.md")
    open(path, "w").write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\n完整报告：{path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
