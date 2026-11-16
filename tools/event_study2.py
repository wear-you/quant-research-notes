#!/usr/bin/env python3
"""event_study2.py — 多批次事件研究：剔除市场 beta 后，业绩预告还有没有漂移？

改进点（对第一版的三个修正）：
  1. **跨越多个季度**——单批次可能只是那段时间的市场风格
  2. **剔除市场收益**——用沪深300 同期收益做基准，算超额，否则测到的是 beta
  3. **给出标准误与 t 值**——单看均值会被极端值骗

口径：
  - 公告通常在收盘后 → 基准 = 公告日之后第一个交易日收盘
  - 超额 = 个股 N 日收益 − 沪深300 同期收益
  - 价格本地缓存，重复运行不重复抓
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
OUT = os.path.join(FIN, "company", "event_study.json")
H = (5, 20)

QUARTERS = ["20230331", "20230630", "20230930", "20231231",
            "20240331", "20240630", "20240930", "20241231",
            "20250331", "20250630", "20250930",
            "20260331", "20260630", "20260930"]
GROUPS = ["预增", "预减"]
PER = 25

_px = {}


def load_cache():
    global _px
    if os.path.exists(CACHE):
        try:
            _px = json.load(open(CACHE))
        except Exception:
            _px = {}


def save_cache():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(_px, open(CACHE, "w"))


def get_prices(code):
    if code in _px:
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
    _px[code] = []
    return []


def bench():
    p = os.path.join(FIN, "data", "csi300.csv")
    return [[r["date"], float(r["close"])] for r in csv.DictReader(open(p))]


def window_ret(px, start_idx, h):
    if start_idx + h >= len(px):
        return None
    return px[start_idx + h][1] / px[start_idx][1] - 1


def market_ret(bx, d0, d1):
    """沪深300 在 d0→d1 之间的收益。"""
    def close_on_or_after(d):
        for x in bx:
            if x[0] >= d:
                return x[1]
        return None
    a, b = close_on_or_after(d0), close_on_or_after(d1)
    return (b / a - 1) if (a and b) else None


def study():
    load_cache()
    bx = bench()
    rows = []
    for q in QUARTERS:
        try:
            df = ak.stock_yjyg_em(date=q)
        except Exception as e:
            print(f"{q} 预告拉取失败 {type(e).__name__}")
            continue
        if df is None or len(df) == 0:
            continue
        for g in GROUPS:
            sub = df[df["预告类型"] == g]
            if len(sub) == 0:
                continue
            sub = sub.copy()
            sub["code"] = sub["股票代码"].astype(str).str.zfill(6)
            sub = sub.drop_duplicates("code").head(PER)
            got = 0
            for _, r in sub.iterrows():
                code, ann = r["code"], str(r["公告日期"])[:10]
                px = get_prices(code)
                if len(px) < 60:
                    continue
                idx = next((i for i, x in enumerate(px) if x[0] > ann), None)
                if idx is None or idx < 1 or idx + max(H) >= len(px):
                    continue
                row = {"q": q, "type": g, "code": code, "ann": ann}
                ok = False
                for h in H:
                    sr = window_ret(px, idx, h)
                    if sr is None:
                        continue
                    mr = market_ret(bx, px[idx][0], px[idx + h][0])
                    row[f"r{h}"] = sr
                    row[f"adj{h}"] = (sr - mr) if mr is not None else None
                    ok = True
                if ok:
                    rows.append(row)
                    got += 1
            save_cache()
            print(f"  {q} {g}: {got} 只")

    def agg(key, sel):
        v = [x[key] for x in rows if sel(x) and x.get(key) is not None]
        if len(v) < 5:
            return None
        m = statistics.mean(v)
        se = statistics.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0
        return {"n": len(v), "mean": m, "se": se,
                "t": m / se if se else 0,
                "win": sum(1 for z in v if z > 0) / len(v)}

    print(f"\n总样本 {len(rows)} 条\n")
    print(f"{'类型':6s} {'n':>5s} {'超额+5d':>10s} {'t':>6s} {'胜率':>7s} "
          f"{'超额+20d':>10s} {'t':>6s} {'胜率':>7s}")
    print("-" * 62)
    summary = {}
    for g in GROUPS:
        a5 = agg("adj5", lambda x, g=g: x["type"] == g)
        a20 = agg("adj20", lambda x, g=g: x["type"] == g)
        if not a5:
            continue
        summary[g] = {"adj5": a5, "adj20": a20}
        print(f"{g:6s} {a5['n']:5d} {a5['mean']*100:+9.2f}% {a5['t']:6.2f} "
              f"{a5['win']*100:6.1f}% "
              f"{(a20['mean']*100 if a20 else 0):+9.2f}% {(a20['t'] if a20 else 0):6.2f} "
              f"{(a20['win']*100 if a20 else 0):6.1f}%")

    if all(g in summary for g in GROUPS):
        d5 = summary["预增"]["adj5"]["mean"] - summary["预减"]["adj5"]["mean"]
        d20 = summary["预增"]["adj20"]["mean"] - summary["预减"]["adj20"]["mean"]
        print(f"\n预增 − 预减：+5日 {d5*100:+.2f}pp，+20日 {d20*100:+.2f}pp")

    # 分季度看符号是否一致
    print(f"\n{'季度':10s} {'预增超额5d':>12s} {'n':>4s} {'预减超额5d':>12s} {'n':>4s}")
    print("-" * 46)
    signs = []
    for q in QUARTERS:
        line = f"{q:10s}"
        vals = {}
        for g in GROUPS:
            v = [x["adj5"] for x in rows if x["q"] == q and x["type"] == g
                 and x.get("adj5") is not None]
            vals[g] = (statistics.mean(v), len(v)) if len(v) >= 5 else None
        for g in GROUPS:
            if vals[g]:
                line += f" {vals[g][0]*100:+11.2f}% {vals[g][1]:4d}"
            else:
                line += f" {'—':>11s} {0:4d}"
        print(line)
        if vals["预增"] and vals["预减"]:
            signs.append(vals["预增"][0] - vals["预减"][0])
    if signs:
        neg = sum(1 for s in signs if s < 0)
        print(f"\n预增跑输预减的季度：{neg}/{len(signs)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"rows": rows, "summary": summary,
               "generated": datetime.now().isoformat()},
              open(OUT, "w"), ensure_ascii=False, default=str)
    print(f"\n明细: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(study())
