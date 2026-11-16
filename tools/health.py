#!/usr/bin/env python3
"""health.py — 金融流水线自检。

回答一个问题：这套系统今天还能不能信？

检查项：
  A. 数据   每个序列末条日期、根数、是否过期
  B. 账本   重复键、净值非正、仓位越界、日期缺口
  C. 参数   冻结参数是否被偷偷改动（与 params.lock.json 比对）
  D. 运行   最近一次成功运行距今多久

输出：~/finance/health/latest.json + latest.md
退出码：0 = 全部通过或仅警告；1 = 有 FAIL（cron 会因此报警）
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta

FIN = os.path.expanduser("~/finance")
DATA = os.path.join(FIN, "data")
LEDGER = os.path.join(FIN, "ledger.csv")
LOCK = os.path.join(FIN, "params.lock.json")
OUT = os.path.join(FIN, "health")

STALE_DAYS = 10
MIN_BARS = 300
MAX_LEDGER_GAP_DAYS = 10

# 冻结参数（与 paper.py 对照）
FROZEN = {"WINDOW": 100, "TARGET_VOL": 0.10, "CASH_RATE": 0.015}


def parse_consts(path):
    src = open(path, encoding="utf-8").read()
    out = {}
    for k in ("WINDOW", "TARGET_VOL", "CASH_RATE"):
        m = re.search(rf"^{k}\s*=\s*([0-9.]+)", src, re.M)
        if m:
            out[k] = float(m.group(1))
    return out, src


def main():
    os.makedirs(OUT, exist_ok=True)
    now = datetime.now()
    fails, warns, oks = [], [], []

    # ---- A. 数据 ----
    files = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    stale = []
    total_bars = 0
    for p in files:
        name = os.path.basename(p)[:-4]
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception as e:
            fails.append(f"数据 {name} 读取失败：{e}")
            continue
        if not rows:
            fails.append(f"数据 {name} 为空")
            continue
        total_bars += len(rows)
        last = rows[-1].get("date", "")
        try:
            age = (now.date() - datetime.strptime(last, "%Y-%m-%d").date()).days
        except ValueError:
            fails.append(f"数据 {name} 日期格式异常：{last}")
            continue
        if age > STALE_DAYS:
            stale.append((name, last, age))
        if len(rows) < MIN_BARS:
            warns.append(f"数据 {name} 只有 {len(rows)} 根")

    if stale:
        fails.append("过期序列 " + "、".join(f"{n}(末{ d })" for n, d, _ in stale))
    else:
        oks.append(f"数据 {len(files)} 个序列全部在 {STALE_DAYS} 天内，共 {total_bars:,} 根")

    # ---- B. 账本 ----
    if not os.path.exists(LEDGER):
        warns.append("账本不存在（今天还没跑）")
    else:
        rows = list(csv.DictReader(open(LEDGER)))
        keys, dup = set(), 0
        bad_eq, bad_pos = 0, 0
        dates = set()
        if rows:
            header = rows[0].keys()
            if "equity" not in header or "pos" not in header:
                fails.append(f"账本列缺失：{list(header)}")
        for r in rows:
            k = (r["date"], r["asset"], r["strategy"])
            if k in keys:
                dup += 1
            keys.add(k)
            dates.add(r["date"])
            try:
                if float(r["equity"]) <= 0:
                    bad_eq += 1
                p = float(r["pos"])
                if p < -1e-9 or p > 1 + 1e-9:
                    bad_pos += 1
            except (ValueError, KeyError):
                bad_pos += 1
        if dup:
            fails.append(f"账本重复键 {dup} 条")
        if bad_eq:
            fails.append(f"账本净值非正 {bad_eq} 条")
        if bad_pos:
            fails.append(f"账本仓位越界 {bad_pos} 条")
        ds = sorted(dates)
        if len(ds) >= 2:
            gap = (datetime.strptime(ds[-1], "%Y-%m-%d") -
                   datetime.strptime(ds[-2], "%Y-%m-%d")).days
            if gap > MAX_LEDGER_GAP_DAYS:
                warns.append(f"账本最近两次记录相隔 {gap} 天")
        if not (dup or bad_eq or bad_pos):
            oks.append(f"账本 {len(rows)} 行、{len(dates)} 个交易日，无重复/越界")

    # ---- C. 参数锁 ----
    paper = os.path.join(FIN, "paper.py")
    if not os.path.exists(paper):
        fails.append("paper.py 不存在")
    else:
        cur, src = parse_consts(paper)
        h = hashlib.sha256(
            json.dumps({k: cur.get(k) for k in FROZEN}, sort_keys=True).encode()
        ).hexdigest()[:16]
        if not os.path.exists(LOCK):
            json.dump({**FROZEN, "sha": h, "locked_at": now.isoformat(timespec="seconds")},
                      open(LOCK, "w"), ensure_ascii=False, indent=2)
            oks.append(f"参数锁已建立 {cur} sha={h}")
        else:
            lock = json.load(open(LOCK))
            drift = {k: (lock.get(k), cur.get(k)) for k in FROZEN
                     if abs(float(lock.get(k, -1)) - float(cur.get(k, -2))) > 1e-12}
            if drift:
                fails.append("冻结参数被改动：" +
                             "、".join(f"{k} {a}→{b}" for k, (a, b) in drift.items()))
            else:
                oks.append(f"参数未改动 {cur} sha={h}")

    # ---- D. 最近运行 ----
    rep = os.path.join(FIN, "reports")
    last_run = None
    if os.path.isdir(rep):
        fs = [os.path.getmtime(os.path.join(rep, f)) for f in os.listdir(rep)]
        if fs:
            last_run = datetime.fromtimestamp(max(fs))
    if last_run is None:
        warns.append("没有任何日报（流水线可能从未跑过）")
    else:
        hrs = (now - last_run).total_seconds() / 3600
        if hrs > 48:
            fails.append(f"最近一次日报在 {hrs:.0f} 小时前")
        else:
            oks.append(f"最近日报 {last_run:%Y-%m-%d %H:%M}（{hrs:.1f} 小时前）")

    # ---- E. 预测哈希链 ----
    pred = os.path.join(FIN, "predictions.jsonl")
    if os.path.exists(pred):
        try:
            recs = [json.loads(l) for l in open(pred) if l.strip()]
            broken = None
            for i, r in enumerate(recs):
                if i == 0:
                    if r.get("prev") != "GENESIS":
                        broken = i
                        break
                else:
                    h = hashlib.sha256(
                        json.dumps(recs[i - 1], ensure_ascii=False,
                                   sort_keys=True).encode()).hexdigest()
                    if r.get("prev") != h:
                        broken = i
                        break
            if broken is not None:
                fails.append(f"预测记录哈希链在第 {broken+1} 条断裂（可能被改写）")
            else:
                oks.append(f"预测记录 {len(recs)} 条，哈希链完整")
        except Exception as e:
            fails.append(f"预测记录解析失败：{e}")

    # ---- 汇总 ----
    status = "FAIL" if fails else ("WARN" if warns else "PASS")
    lines = [f"# 金融流水线自检 · {now:%Y-%m-%d %H:%M}", "", f"**状态：{status}**", ""]
    for title, items in (("❌ 失败", fails), ("⚠️ 警告", warns), ("✅ 通过", oks)):
        if items:
            lines.append(f"## {title}")
            lines += [f"- {i}" for i in items]
            lines.append("")

    open(os.path.join(OUT, "latest.md"), "w").write("\n".join(lines))
    json.dump({"status": status, "fails": fails, "warns": warns, "oks": oks,
               "checked_at": now.isoformat(timespec="seconds")},
              open(os.path.join(OUT, "latest.json"), "w"), ensure_ascii=False, indent=2)

    print("\n".join(lines))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
