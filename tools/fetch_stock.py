#!/usr/bin/env python3
"""fetch_stock.py — 拉单只标的（个股/ETF）日线，合并进 data/。

用法：python3 fetch_stock.py moutai sh600519
"""
import csv
import os
import sys

import akshare as ak

DATA = os.path.expanduser("~/finance/data")
FIELDS = ["date", "open", "high", "low", "close", "volume"]


def main():
    if len(sys.argv) < 3:
        print("用法: fetch_stock.py <名称> <新浪符号，如 sh600519>")
        return 2
    name, sym = sys.argv[1], sys.argv[2]
    path = os.path.join(DATA, f"{name}.csv")

    merged = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            merged[r["date"]] = {k: r[k] for k in FIELDS}

    df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
    for _, r in df.iterrows():
        d = str(r["date"])[:10]
        merged[d] = {"date": d, "open": f"{r['open']:.4f}", "high": f"{r['high']:.4f}",
                     "low": f"{r['low']:.4f}", "close": f"{r['close']:.4f}",
                     "volume": f"{r['volume']:.0f}"}

    out = [merged[k] for k in sorted(merged)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"{name} ({sym}) {len(out)} 根  {out[0]['date']} → {out[-1]['date']}  收盘 {out[-1]['close']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
