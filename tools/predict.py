#!/usr/bin/env python3
"""predict.py — 预测 → 锁定 → 到期验证 → 校准。

为什么这样设计：
  "我能预测"这句话没有任何信息量。有信息量的是**先锁死的概率 + 到期后的分数**。
  所以这里的每条预测都写进带哈希链的 predictions.jsonl，事后不可改；
  到期由 settle 自动结算，score 给出校准表与 Brier 分数。

方法（刻意保守）：
  不做"故事型"预测。只回答一个经验问题——
  「在这个资产历史上，处于当前状态（趋势上/下 × 波动高/低）时，
    之后 H 个交易日上涨的频率是多少？」
  这就是条件经验频率 + 拉普拉斯平滑，没有模型、没有过拟合空间。

用法：
  python3 predict.py auto    # 结算到期 + 周一则新建（cron 用）
  python3 predict.py make    # 强制新建
  python3 predict.py settle  # 只结算
  python3 predict.py score   # 校准表
"""
import csv
import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone, timedelta

DATA = os.path.expanduser("~/finance/data")
PRED = os.path.expanduser("~/finance/predictions.jsonl")
SETTLED = os.path.expanduser("~/finance/predictions_settled.csv")
TD = 244
HORIZON = 20
WINDOW, VOLW = 100, 60
TZ = timezone(timedelta(hours=8))


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))]


def state_samples(prices):
    """返回 [(state, fwd_return)]，state = 'up'/'down' + 'hi'/'lo'。"""
    n = len(prices)
    rets = [0.0] + [prices[i] / prices[i - 1] - 1 for i in range(1, n)]
    vols = [0.0] * n
    for i in range(VOLW, n):
        vols[i] = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5
    valid = [v for v in vols[VOLW:] if v > 0]
    med = statistics.median(valid) if valid else 0

    out = []
    for i in range(WINDOW, n - HORIZON):
        ma = sum(prices[i - WINDOW + 1:i + 1]) / WINDOW
        up = prices[i] > ma
        hi = vols[i] > med
        fwd = prices[i + HORIZON] / prices[i] - 1
        out.append((("up" if up else "down") + ("hi" if hi else "lo"), fwd))
    return out


def current_state(prices):
    ma = sum(prices[-WINDOW:]) / WINDOW
    rets = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - VOLW, len(prices))]
    vol = statistics.pstdev(rets) * TD ** 0.5
    samples = state_samples(prices)
    valid = [v for v in
             (statistics.pstdev([prices[j] / prices[j - 1] - 1
                                 for j in range(i - VOLW + 1, i + 1)]) * TD ** 0.5
              for i in range(VOLW, len(prices)) ) if v > 0]
    med = statistics.median(valid) if valid else 0
    return ("up" if prices[-1] > ma else "down") + ("hi" if vol > med else "lo")


def make_one(name, horizon=HORIZON):
    rows = load(name)
    if len(rows) < WINDOW + VOLW + 50:
        return None
    prices = [p for _, p in rows]
    st = current_state(prices)
    samples = [f for s, f in state_samples(prices) if s == st]
    allf = [f for _, f in state_samples(prices)]
    if len(samples) < 20:
        return None

    n = len(samples)
    ups = sum(1 for f in samples if f > 0)
    p_up = (ups + 0.5) / (n + 1)                      # 拉普拉斯平滑
    base_ups = sum(1 for f in allf if f > 0)
    p_base = (base_ups + 0.5) / (len(allf) + 1)
    med_ret = statistics.median(samples)
    p25 = sorted(samples)[int(0.25 * n)]
    p75 = sorted(samples)[int(0.75 * n)]

    return {"id": hashlib.sha256(
                f"{name}{rows[-1][0]}{datetime.now(TZ).isoformat()}".encode()).hexdigest()[:16],
            "created_at": datetime.now(TZ).isoformat(timespec="seconds"),
            "asset": name, "as_of": rows[-1][0], "price": prices[-1],
            "horizon_days": horizon, "due_date": None,
            "state": st, "n": n,
            "p_up": round(p_up, 4), "p_base": round(p_base, 4),
            "median_return": round(med_ret, 4),
            "q25": round(p25, 4), "q75": round(p75, 4),
            "method": "conditional empirical frequency, Laplace smoothing"}


def read_all():
    if not os.path.exists(PRED):
        return []
    return [json.loads(l) for l in open(PRED) if l.strip()]


def append(rec):
    recs = read_all()
    prev = hashlib.sha256(json.dumps(recs[-1], sort_keys=True).encode()).hexdigest() if recs else "GENESIS"
    rec["prev"] = prev
    with open(PRED, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def settled_ids():
    if not os.path.exists(SETTLED):
        return set()
    return {r["id"] for r in csv.DictReader(open(SETTLED))}


def settle():
    done = settled_ids()
    recs = read_all()
    rows_out = []
    for r in recs:
        if r["id"] in done:
            continue
        rows = load(r["asset"])
        idx = {d: i for i, (d, _) in enumerate(rows)}
        if r["as_of"] not in idx:
            continue
        i0 = idx[r["as_of"]]
        i1 = i0 + r["horizon_days"]
        if i1 >= len(rows):
            continue                                   # 还没到期
        p0, p1 = rows[i0][1], rows[i1][1]
        ret = p1 / p0 - 1
        up = 1 if ret > 0 else 0
        rows_out.append({"id": r["id"], "asset": r["asset"], "as_of": r["as_of"],
                         "settle_date": rows[i1][0], "price0": f"{p0:.4f}",
                         "price1": f"{p1:.4f}", "return": f"{ret:.4f}", "up": up,
                         "p_up": r["p_up"], "brier": f"{(r['p_up'] - up) ** 2:.4f}"})
    if rows_out:
        exists = os.path.exists(SETTLED)
        with open(SETTLED, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            if not exists:
                w.writeheader()
            w.writerows(rows_out)
        print(f"结算 {len(rows_out)} 条")
        for r in rows_out:
            print(f"  {r['asset']:10s} {r['as_of']}→{r['settle_date']} "
                  f"实际 {float(r['return'])*100:+6.2f}%  预测涨概率 {float(r['p_up']):.0%}  "
                  f"{'✔' if (float(r['p_up'])>0.5)==(r['up']=='1') else '✘'}")
    else:
        print("没有到期的预测")
    return rows_out


def score():
    if not os.path.exists(SETTLED):
        print("还没有已结算的预测")
        return
    rows = list(csv.DictReader(open(SETTLED)))
    if not rows:
        print("还没有已结算的预测")
        return
    n = len(rows)
    brier = sum(float(r["brier"]) for r in rows) / n
    wins = sum(1 for r in rows if (float(r["p_up"]) > 0.5) == (r["up"] == "1"))
    print(f"已结算 {n} 条")
    print(f"方向命中率 {wins/n*100:.1f}%")
    print(f"Brier 分数 {brier:.4f}（0.25 = 无信息，越低越好）")
    print("\n校准表：")
    print(f"{'预测区间':12s} {'条数':>4s} {'预测均值':>8s} {'实际上涨率':>10s}")
    buckets = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= float(r["p_up"]) < hi]
        if not sel:
            continue
        pm = sum(float(r["p_up"]) for r in sel) / len(sel)
        ar = sum(1 for r in sel if r["up"] == "1") / len(sel)
        print(f"{lo:.1f}–{hi:.1f}      {len(sel):4d} {pm*100:7.1f}% {ar*100:9.1f}%")
    print("\n解读：预测均值与实际上涨率越接近，校准越好；偏离说明概率不可信。")


def auto(assets):
    print("=== 结算 ===")
    settle()
    print("\n=== 新建 ===")
    today = datetime.now(TZ)
    existing = {r["asset"] for r in read_all() if r["id"] not in settled_ids()}
    if today.weekday() != 0:
        print("今天不是周一，跳过新建（每周一预测一次，20 日窗口）")
        return
    for a in assets:
        if a in existing:
            print(f"{a}: 已有未结算预测，跳过")
            continue
        rec = make_one(a)
        if not rec:
            print(f"{a}: 样本不足")
            continue
        append(rec)
        print(f"{a}: 已锁定 {rec['state']}  p_up={rec['p_up']:.2f}  "
              f"中位 {rec['median_return']*100:+.2f}%  n={rec['n']}  id={rec['id']}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    assets = ["moutai", "csi300"]
    if cmd == "auto":
        auto(assets)
    elif cmd == "make":
        for a in assets:
            rec = make_one(a)
            if rec:
                append(rec)
                print(f"{a}: 已锁定 {rec['state']} p_up={rec['p_up']:.2f} id={rec['id']}")
    elif cmd == "settle":
        settle()
    else:
        score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
