#!/usr/bin/env python3
"""check_data.py — 数据完整性闸门。

背景：akshare 对某些符号会返回**过期快照**（例如 csi_div 停在 2019-01），
如果不过滤，回测会拿一段早就结束的历史当成"最近表现"，悄无声息地毁掉结论。

规则：
  1. 末尾日期距今超过 STALE_DAYS 天 → 判过期，移入 data/_quarantine/
  2. 根数 < MIN_BARS → 判样本不足，同样隔离
  3. 中间有 > 20 个交易日的缺口 → 只警告，不隔离
"""
import csv
import os
import shutil
import sys
from datetime import datetime, timedelta

DATA = os.path.expanduser("~/finance/data")
QDIR = os.path.join(DATA, "_quarantine")
STALE_DAYS = 10
MIN_BARS = 300


def load_dates(path):
    rows = list(csv.DictReader(open(path)))
    return [r["date"] for r in rows], rows


def main():
    os.makedirs(QDIR, exist_ok=True)
    today = datetime.now().date()
    ok, stale, short = [], [], []

    for fn in sorted(os.listdir(DATA)):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(DATA, fn)
        try:
            dates, rows = load_dates(path)
        except Exception as e:
            print(f"{fn:20s} 读取失败 {e}")
            continue
        if not dates:
            short.append((fn, 0, "", ""))
            continue
        first, last = dates[0], dates[-1]
        try:
            last_d = datetime.strptime(last, "%Y-%m-%d").date()
        except ValueError:
            stale.append((fn, len(dates), last, "日期格式异常"))
            continue

        age = (today - last_d).days
        if age > STALE_DAYS:
            stale.append((fn, len(dates), last, f"末尾距今 {age} 天"))
        elif len(dates) < MIN_BARS:
            short.append((fn, len(dates), first, last))
        else:
            ok.append((fn, len(dates), first, last))

    print(f"=== 健康 {len(ok)} 个 ===")
    for fn, n, f, l in ok:
        print(f"  {fn:20s} {n:5d} 根  {f} → {l}")

    if stale:
        print(f"\n=== 过期/异常 {len(stale)} 个 → 隔离 ===")
        for fn, n, last, why in stale:
            src = os.path.join(DATA, fn)
            dst = os.path.join(QDIR, fn)
            shutil.move(src, dst)
            print(f"  {fn:20s} {n:5d} 根  末 {last}  {why}  → _quarantine/")

    if short:
        print(f"\n=== 样本不足 {len(short)} 个 → 隔离 ===")
        for fn, n, f, l in short:
            src = os.path.join(DATA, fn)
            if os.path.exists(src):
                shutil.move(src, os.path.join(QDIR, fn))
            print(f"  {fn:20s} {n:5d} 根  {f} → {l}")

    # 缺口检测（只警告）
    print("\n=== 缺口警告（>20 天，仅提示）===")
    any_gap = False
    for fn, n, f, l in ok:
        try:
            ds = [datetime.strptime(d, "%Y-%m-%d").date()
                  for d in load_dates(os.path.join(DATA, fn))[0]]
        except ValueError:
            continue
        gaps = [(ds[i - 1], ds[i], (ds[i] - ds[i - 1]).days)
                for i in range(1, len(ds)) if (ds[i] - ds[i - 1]).days > 20]
        if gaps:
            any_gap = True
            a, b, g = max(gaps, key=lambda x: x[2])
            print(f"  {fn:20s} 最大缺口 {g} 天  {a} → {b}")
    if not any_gap:
        print("  无")
    return 0


if __name__ == "__main__":
    sys.exit(main())
