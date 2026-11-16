#!/usr/bin/env python3
"""yjyg_paper.py — 业绩预告「预增」5 日漂移策略的前向验证。

后视问题：事件研究是在已知全部历史的前提下算的，无法排除数据窥探。
本脚本改用**真实时间轴**：每天扫描新公告，按规则记下入场点，等时间到了再结算。

规则（写死，不许改）：
  - 只做「预增」
  - 入场：公告日之后第一个交易日收盘
  - 出场：入场后第 5 个交易日收盘
  - 等权，每笔 1 单位；不计成本（成本单列，用于敏感性）
  - 超额 = 个股收益 − 沪深300 同期收益

状态存 company/yjyg_ledger.csv，新增/结算都追加，不覆盖历史。
"""
import csv
import json
import os
import statistics
import sys
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
import akshare as ak

FIN = os.path.expanduser("~/finance")
DIR = os.path.join(FIN, "company")
LEDGER = os.path.join(DIR, "yjyg_ledger.csv")
CACHE = os.path.join(FIN, "cache", "px.json")
TZ = timezone(timedelta(hours=8))
HOLD = 5
LOOKBACK = 12          # 扫描最近多少天发布的预告
COLS = ["code", "name", "ann_date", "entry_date", "entry_px",
        "exit_date", "exit_px", "ret", "bench_ret", "adj_ret", "status"]

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
        if df is not None and len(df) >= 1:
            _px[code] = [[str(r["date"])[:10], float(r["close"])] for _, r in df.iterrows()]
            return _px[code]
    except Exception:
        pass
    # 失败不缓存，避免一次网络抖动把某只股票永久判死
    return []


def calendar():
    p = os.path.join(FIN, "data", "csi300.csv")
    return [r["date"] for r in csv.DictReader(open(p))]


def bench_px():
    p = os.path.join(FIN, "data", "csi300.csv")
    return {r["date"]: float(r["close"]) for r in csv.DictReader(open(p))}


def read_ledger():
    if not os.path.exists(LEDGER):
        return []
    return list(csv.DictReader(open(LEDGER)))


def write_ledger(rows):
    os.makedirs(DIR, exist_ok=True)
    with open(LEDGER, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def recent_periods(today, n=4):
    """返回最近 n 个季度末日期字符串。"""
    q = [(3, 31), (6, 30), (9, 30), (12, 31)]
    out, y = [], today.year
    while len(out) < n:
        for m, d in reversed(q):
            dt = datetime(y, m, d).date()
            if dt <= today:
                out.append(dt.strftime("%Y%m%d"))
            if len(out) >= n:
                break
        y -= 1
    return out


def main():
    load_cache()
    cal = calendar()
    bpx = bench_px()
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    rows = read_ledger()
    have = {(r["code"], r["ann_date"]) for r in rows}

    # 1. 扫描新公告
    cut = (datetime.now(TZ).date() - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")
    new = 0
    for per in recent_periods(datetime.now(TZ).date()):
        try:
            df = ak.stock_yjyg_em(date=per)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        sub = df[df["预告类型"] == "预增"].copy()
        sub["code"] = sub["股票代码"].astype(str).str.zfill(6)
        sub["ann"] = sub["公告日期"].astype(str).str[:10]
        for _, r in sub.iterrows():
            if r["ann"] < cut or (r["code"], r["ann"]) in have:
                continue
            rows.append({"code": r["code"], "name": r["股票简称"],
                         "ann_date": r["ann"], "entry_date": "", "entry_px": "",
                         "exit_date": "", "exit_px": "", "ret": "", "bench_ret": "",
                         "adj_ret": "", "status": "pending"})
            have.add((r["code"], r["ann"]))
            new += 1

    # 2. 补入场价
    filled = 0
    for r in rows:
        if r["status"] != "pending" or r["entry_date"]:
            continue
        px = get_prices(r["code"])
        if not px:
            continue
        idx = next((i for i, x in enumerate(px) if x[0] > r["ann_date"]), None)
        if idx is None:
            continue
        if idx == 0:
            # 公告发出时尚未上市，不可交易
            r["status"] = "invalid"
            continue
        r["entry_date"] = px[idx][0]
        r["entry_px"] = px[idx][1]
        r["status"] = "open"
        filled += 1

    # 3. 结算到期
    settled = 0
    for r in rows:
        if r["status"] != "open":
            continue
        px = get_prices(r["code"])
        if not px:
            continue
        ds = [x[0] for x in px]
        if r["entry_date"] not in ds:
            continue
        i = ds.index(r["entry_date"])
        if i + HOLD >= len(px):
            continue
        r["exit_date"] = px[i + HOLD][0]
        r["exit_px"] = px[i + HOLD][1]
        r["ret"] = round(px[i + HOLD][1] / px[i][1] - 1, 6)
        b0, b1 = bpx.get(r["entry_date"]), bpx.get(r["exit_date"])
        if b0 and b1:
            r["bench_ret"] = round(b1 / b0 - 1, 6)
            r["adj_ret"] = round(float(r["ret"]) - float(r["bench_ret"]), 6)
        r["status"] = "closed"
        settled += 1

    save_cache()
    write_ledger(rows)

    done = [r for r in rows if r["status"] == "closed" and r["adj_ret"] not in ("", None)]
    print(f"新信号 {new}  补入场 {filled}  结算 {settled}")
    print(f"账本条数 {len(rows)}  已结算 {len(done)}  待入场 "
          f"{sum(1 for r in rows if r['status']=='pending')}  持仓中 "
          f"{sum(1 for r in rows if r['status']=='open')}")
    if done:
        v = [float(r["adj_ret"]) for r in done]
        raw = [float(r["ret"]) for r in done]
        print(f"\n已结算 {len(v)} 笔")
        print(f"  平均原始收益 {statistics.mean(raw)*100:+.2f}%")
        print(f"  平均超额     {statistics.mean(v)*100:+.2f}%")
        print(f"  胜率(超额>0) {sum(1 for x in v if x>0)/len(v)*100:.1f}%")
        if len(v) > 2:
            se = statistics.stdev(v) / len(v) ** 0.5
            print(f"  t = {statistics.mean(v)/se:.2f}（未做簇稳健，仅供参考）")
        print(f"  扣 25bp 后   {(statistics.mean(v)-0.0025)*100:+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
