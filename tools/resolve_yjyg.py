#!/usr/bin/env python3
"""resolve_yjyg.py — 查清「预增」方向矛盾：全市场 +2.04pp vs 沪深300 内部 −0.0207。

假设：预增的信息量集中在中小盘，沪深300 内部几乎没有。

做法：**同一套协议**，只换股票池。
  - 事件定义：业绩预告「预增」/「预减」
  - 入场：公告日之后第一个交易日收盘
  - 出场：第 5 个交易日收盘
  - 超额 = 个股收益 − 沪深300 同期收益
  - 分组：公告时属于当前沪深300 成分股 / 不属于
  - 统计：每季度聚成一个观测，再对季度做 t 检验（簇稳健）

⚠️ 成分股用的是**当前**名单，不是历史名单 → 这里的"是否属于300"本身有前视偏差，
   结论只能作为"效应是否集中在大盘股"的初步判断，不能当精确的分组归因。
"""
import csv
import json
import os
import statistics
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import akshare as ak

FIN = os.path.expanduser("~/finance")
CACHE = os.path.join(FIN, "cache", "px.json")
OUT = os.path.join(FIN, "company", "resolve_yjyg.json")
H = 5
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 40
QUARTERS = ["20230331", "20230630", "20230930", "20231231",
            "20240331", "20240630", "20240930", "20241231",
            "20250331", "20250630", "20250930",
            "20260331", "20260630", "20260930"]
TYPES = ["预增", "预减"]

_px = {}


def load_cache():
    global _px
    if os.path.exists(CACHE):
        try:
            _px = json.load(open(CACHE))
        except Exception:
            print("缓存无法读取，从空开始")
            _px = {}


def save_cache():
    tmp = f"{CACHE}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_px, f)
    os.replace(tmp, CACHE)


def get_prices(code):
    if code in _px and _px[code]:
        return _px[code]
    sym = f"sh{code}" if code.startswith(("6", "9")) else (
        f"bj{code}" if code.startswith(("4", "8")) else f"sz{code}")
    try:
        df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
        if df is not None and len(df) > 60:
            _px[code] = [[str(r["date"])[:10], float(r["close"])] for _, r in df.iterrows()]
            return _px[code]
    except Exception:
        pass
    return []


def bench_map():
    p = os.path.join(FIN, "data", "csi300.csv")
    return {r["date"]: float(r["close"]) for r in csv.DictReader(open(p))}


def csi300_set():
    df = ak.index_stock_cons_csindex(symbol="000300")
    return {str(c).zfill(6) for c in df["成分券代码"]}


def cluster_stats(by_quarter):
    vals = [v for v in by_quarter.values() if v is not None]
    if len(vals) < 4:
        return None
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    return {"nq": len(vals), "mean": m, "t": m / (sd / len(vals) ** 0.5) if sd else 0,
            "pos": sum(1 for v in vals if v > 0)}


def main():
    load_cache()
    bx = bench_map()
    c300 = csi300_set()
    print(f"沪深300 成分 {len(c300)} 只\n")

    # results[(type, group)][quarter] = [exc...]
    res = {}
    for q in QUARTERS:
        try:
            df = ak.stock_yjyg_em(date=q)
        except Exception as e:
            print(f"{q} 失败 {type(e).__name__}")
            continue
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["code"] = df["股票代码"].astype(str).str.zfill(6)
        df["ann"] = df["公告日期"].astype(str).str[:10]
        for typ in TYPES:
            sub = df[df["预告类型"] == typ].drop_duplicates("code")
            if len(sub) == 0:
                continue
            # 每组按 300 / 非300 各取 PER 只
            for grp, cond in (("HS300", True), ("NON300", False)):
                pool = sub[sub["code"].isin(c300) == cond].head(PER)
                got = 0
                for _, r in pool.iterrows():
                    px = get_prices(r["code"])
                    if len(px) < 60:
                        continue
                    ds = [x[0] for x in px]
                    idx = next((i for i, d in enumerate(ds) if d > r["ann"]), None)
                    if idx is None or idx < 1 or idx + H >= len(px):
                        continue
                    r0, r1 = px[idx][0], px[idx + H][0]
                    b0, b1 = bx.get(r0), bx.get(r1)
                    if not (b0 and b1):
                        continue
                    exc = (px[idx + H][1] / px[idx][1] - 1) - (b1 / b0 - 1)
                    res.setdefault((typ, grp), {}).setdefault(q, []).append(exc)
                    got += 1
                if got:
                    print(f"  {q} {typ} {grp}: {got} 只")
        save_cache()
    save_cache()

    print("\n" + "=" * 78)
    print("同一协议，只换股票池：5 日超额收益（季度聚簇稳健）")
    print("=" * 78)
    print(f"{'类型':6s} {'股票池':8s} {'笔数':>6s} {'季度':>5s} {'均值超额':>10s} "
          f"{'t':>7s} {'正的季度':>9s}")
    print("-" * 78)
    summary = {}
    for typ in TYPES:
        for grp in ("HS300", "NON300"):
            key = (typ, grp)
            if key not in res:
                continue
            bq = {q: statistics.mean(v) for q, v in res[key].items() if len(v) >= 3}
            st = cluster_stats(bq)
            if not st:
                continue
            n = sum(len(v) for v in res[key].values())
            summary[key] = st
            print(f"{typ:6s} {grp:8s} {n:6d} {st['nq']:5d} {st['mean']*100:+9.2f}% "
                  f"{st['t']:+7.2f} {st['pos']:4d}/{st['nq']:<4d}")

    # 直接对照：预增在 300 内外的差
    if ("预增", "HS300") in summary and ("预增", "NON300") in summary:
        a = summary[("预增", "HS300")]["mean"]
        b = summary[("预增", "NON300")]["mean"]
        print(f"\n预增：非300 − 300 = {(b-a)*100:+.2f}pp")
        print("→ 若该差值为正且显著，说明效应集中在中小盘，与 IC 为负不矛盾。")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()},
              open(OUT, "w"), ensure_ascii=False, default=str)
    print(f"\n明细 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
