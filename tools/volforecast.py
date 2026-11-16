#!/usr/bin/env python3
"""volforecast.py — 预测「未来的波动」，而不是「未来的收益」。

为什么换目标：
  已验证两条路都失败：P(涨) 技巧分数 −0.03~−0.07；收益分位数 −0.005~−0.05。
  原因很简单——20 日收益的均值差只有 1.6pp，而波动是 20pp，差 0.08 个标准差，
  任何方法都测不出来。

  但波动率本身是**有记忆的**（波动聚集），这是金融里最稳的实证规律之一。
  而且它不需要预测就能用——我们验证过的策略（目标波动 10%）用的正是它。

验收：
  MAE / RMSE  vs  两种基线（无条件均值、随机游走=用当前波动当预测）
  R²          相对无条件均值的解释力
  方向命中率   预测"波动上升/下降"是否正确
"""
import csv
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
TD = 244
H = 20
WINDOW, VOLW = 100, 60


def load(name):
    p = os.path.join(DATA, f"{name}.csv")
    if not os.path.exists(p):
        return []
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))
            if float(r["close"]) > 0]


def realized_vol(rets, i, h):
    seg = rets[i:i + h]
    if len(seg) < 5:
        return None
    return statistics.pstdev(seg) * TD ** 0.5


def build(rows):
    px = [p for _, p in rows]
    dates = [d for d, _ in rows]
    n = len(px)
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, n)]
    cur = [0.0] * n
    for i in range(VOLW, n):
        cur[i] = statistics.pstdev(rets[i - VOLW + 1:i + 1]) * TD ** 0.5

    recs = []
    for i in range(WINDOW, n - H):
        f = realized_vol(rets, i + 1, H)          # 未来 H 日实现波动
        if f is None or cur[i] <= 0:
            continue
        recs.append({"i": i, "date": dates[i], "cur": cur[i], "future": f})
    return recs, rets


def replay(recs):
    """严格因果：预测 t 时只用 j+H <= t 的样本估计无条件均值。"""
    out, ptr, hist = [], 0, []
    for k, r in enumerate(recs):
        while ptr < k and recs[ptr]["i"] + H <= r["i"]:
            hist.append(recs[ptr]["future"])
            ptr += 1
        if len(hist) < 30:
            continue
        mean_f = statistics.mean(hist)
        prev_cur = recs[k - 1]["cur"] if k > 0 else r["cur"]
        out.append({**r, "base_mean": mean_f, "rw": r["cur"],
                    "prev_cur": prev_cur})
    return out


def report(name, preds):
    if len(preds) < 200:
        print(f"{name:10s} 样本不足")
        return None
    n = len(preds)
    mae = lambda key: sum(abs(p["future"] - p[key]) for p in preds) / n
    rmse = lambda key: (sum((p["future"] - p[key]) ** 2 for p in preds) / n) ** 0.5
    ss_tot = sum((p["future"] - statistics.mean([q["future"] for q in preds])) ** 2
                 for p in preds)
    ss_res = sum((p["future"] - p["rw"]) ** 2 for p in preds)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0

    dirs = [p for p in preds if p["prev_cur"] > 0]
    hit = sum(1 for p in dirs
              if (p["rw"] > p["prev_cur"]) == (p["future"] > p["prev_cur"])) / len(dirs) if dirs else 0

    print(f"\n{name}  样本 {n}  {preds[0]['date']} → {preds[-1]['date']}")
    print(f"  {'方法':14s} {'MAE':>8s} {'RMSE':>8s}")
    print(f"  {'无条件均值':14s} {mae('base_mean')*100:7.2f}% {rmse('base_mean')*100:7.2f}%")
    print(f"  {'当前波动(随机游走)':14s} {mae('rw')*100:7.2f}% {rmse('rw')*100:7.2f}%")
    print(f"  R²（相对无条件均值）{r2:+.3f}")
    print(f"  波动升降方向命中率 {hit*100:.1f}%")
    return {"name": name, "n": n, "r2": r2, "hit": hit,
            "mae_rw": mae("rw"), "mae_base": mae("base_mean")}


def main():
    names = sys.argv[1:] or ["moutai", "csi300", "sse_comp", "gold", "btc"]
    print("预测目标：未来 20 个交易日的已实现波动率")
    res = []
    for nm in names:
        rows = load(nm)
        if len(rows) < WINDOW + VOLW + 300:
            print(f"{nm}: 数据不足")
            continue
        recs, _ = build(rows)
        r = report(nm, replay(recs))
        if r:
            res.append(r)
    if res:
        print("\n=== 汇总 ===")
        ok = [r for r in res if r["r2"] > 0.1]
        print(f"R² > 0.10（有实质预测力）：{len(ok)}/{len(res)}")
        for r in res:
            flag = "✅" if r["r2"] > 0.3 else ("⚠️" if r["r2"] > 0.1 else "❌")
            print(f"  {flag} {r['name']:10s} R² {r['r2']:+.3f}  "
                  f"MAE {r['mae_rw']*100:.2f}% vs 基准 {r['mae_base']*100:.2f}%  "
                  f"方向 {r['hit']*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
