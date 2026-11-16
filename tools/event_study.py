#!/usr/bin/env python3
"""event_study.py — 公司事件到底有没有"提前量"？

问题：业绩预告公布后，股价是不是已经反应完了？
做法：取业绩预告批次，按预告类型分组，从公告日次日收盘起算 5/20 日收益，
      对比各组均值，并和市场基准（同期全部样本的均值）比较。

关键口径：
  - 公告通常在收盘后发布 → 用公告日**之后**第一个交易日的收盘价为基准
  - 收益 = 基准后第 N 个交易日收盘 / 基准日收盘 - 1
  - 同时报告"公告当日涨跌幅"，看市场是不是当天就反应了

用法：python3 event_study.py [季度末日期 默认 20260930] [每组样本数 默认 35]
"""
import statistics
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import akshare as ak

PERIOD = sys.argv[1] if len(sys.argv) > 1 else "20260930"
PER_GROUP = int(sys.argv[2]) if len(sys.argv) > 2 else 35
H = (5, 20)


def get_prices(code):
    """返回 [(date, close)]，前复权。"""
    for sym in (f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}",
                f"bj{code}"):
        try:
            df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
            if df is not None and len(df) > 60:
                return [(str(r["date"])[:10], float(r["close"])) for _, r in df.iterrows()]
        except Exception:
            continue
    return []


def forward(px, ann_date, h):
    idx = next((i for i, (d, _) in enumerate(px) if d > ann_date), None)
    if idx is None or idx + h >= len(px):
        return None, None
    base = px[idx][1]
    return px[idx + h][1] / base - 1, px[idx][1] / px[idx - 1][1] - 1


def main():
    print(f"业绩预告批次 {PERIOD}，每组最多 {PER_GROUP} 只")
    df = ak.stock_yjyg_em(date=PERIOD)
    print(f"全市场预告 {len(df)} 条，类型分布：")
    print(df["预告类型"].value_counts().to_string())

    groups = ["预增", "预减", "扭亏", "首亏", "略增", "略减"]
    stats = {}
    all5, all20 = [], []

    for g in groups:
        sub = df[df["预告类型"] == g]
        if len(sub) == 0:
            continue
        codes = sub["股票代码"].astype(str).str.zfill(6).tolist()[:PER_GROUP]
        r5, r20, day0 = [], [], []
        for c in codes:
            px = get_prices(c)
            if not px:
                continue
            dates = [d for d, _ in px]
            ann = str(sub[sub["股票代码"].astype(str).str.zfill(6) == c]["公告日期"].iloc[0])[:10]
            a5, d0 = forward(px, ann, H[0])
            a20, _ = forward(px, ann, H[1])
            if a5 is not None:
                r5.append(a5)
                day0.append(d0)
                all5.append(a5)
            if a20 is not None:
                r20.append(a20)
                all20.append(a20)
        if len(r5) >= 5:
            stats[g] = {"n": len(r5),
                        "d0": statistics.mean(day0),
                        "r5": statistics.mean(r5),
                        "r20": statistics.mean(r20) if r20 else None,
                        "win5": sum(1 for x in r5 if x > 0) / len(r5)}
        print(f"  {g}: 有效样本 {len(r5)}")

    print(f"\n{'类型':6s} {'样本':>4s} {'公告当日':>9s} {'+5日':>8s} {'+20日':>8s} {'5日胜率':>8s}")
    print("-" * 50)
    for g, s in stats.items():
        r20 = f"{s['r20']*100:+7.2f}%" if s["r20"] is not None else "     —"
        print(f"{g:6s} {s['n']:4d} {s['d0']*100:+8.2f}% {s['r5']*100:+7.2f}% "
              f"{r20} {s['win5']*100:7.1f}%")

    if all5:
        print(f"\n全样本均值：+5日 {statistics.mean(all5)*100:+.2f}%，"
              f"+20日 {statistics.mean(all20)*100:+.2f}%" if all20 else "")
        if "预增" in stats and "预减" in stats:
            spread5 = stats["预增"]["r5"] - stats["预减"]["r5"]
            print(f"预增 − 预减 的 5 日差：{spread5*100:+.2f}pp")
            print("→ 若差值为正且远大于当日涨幅，说明公告后仍有漂移；")
            print("  若差值接近 0，说明市场当天就反应完了，我们拿不到提前量。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
