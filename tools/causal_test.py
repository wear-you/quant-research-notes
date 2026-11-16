#!/usr/bin/env python3
"""causal_test.py — 对入围变量做严格因果的滚动检验。

为什么要单独这一个脚本：
  factors.py 的"样本外 MAE 改善"只切了一次（前 70% 训练 / 后 30% 测试），
  而且对水平变量来说，一次切分很容易被趋势共动骗过去。
  这里改成**逐块向前滚动**：每一块只用它之前的数据拟合，预测完再前进。

指标：
  OOS MAE 改善（相对只含 log(当前波动) 的基线）
  预测序列与真实波动率的相关系数
  方向命中率
"""
import csv
import math
import os
import statistics
import sys

DATA = os.path.expanduser("~/finance/data")
MACRO = os.path.join(DATA, "macro")
TD = 244
H = 20
BLOCK = 60          # 每 60 个交易日重新拟合一次
MIN_TRAIN = 500     # 至少 500 个训练点才开始


def load_series(path, dcol="date", vcol="value"):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append((r[dcol], float(r[vcol])))
        except (KeyError, ValueError):
            continue
    return sorted(out)


def load_csi():
    return [(r["date"], float(r["close"])) for r in csv.DictReader(open(os.path.join(DATA, "csi300.csv")))]


def align(dates, series):
    out, j, last = [], 0, None
    for d in dates:
        while j < len(series) and series[j][0] <= d:
            last = series[j][1]
            j += 1
        out.append(last)
    return out


def ols(X, y):
    k = len(X[0])
    A = [[sum(X[n][i] * X[n][j] for n in range(len(X))) for j in range(k)] +
         [sum(X[n][i] * y[n] for n in range(len(X)))] for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-12:
            return None
        A[i], A[p] = A[p], A[i]
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                for c in range(i, k + 1):
                    A[r][c] -= f * A[i][c]
    return [A[i][k] / A[i][i] for i in range(k)]


def pred(beta, row):
    return sum(b * x for b, x in zip(beta, row))


def pearson(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / n)
    sb = math.sqrt(sum((y - mb) ** 2 for y in b) / n)
    if sa == 0 or sb == 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n / (sa * sb)


def main():
    factors = sys.argv[1:] or ["usdcny", "m2", "cn10y", "shibor_3m", "lpr1y", "cn_slope"]
    csi = load_csi()
    dates = [d for d, _ in csi]
    px = [p for _, p in csi]
    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, len(px))]
    cur = [statistics.pstdev(rets[max(0, i - 60):i + 1]) * TD ** 0.5 for i in range(len(px))]
    fwd = [None] * len(px)
    for i in range(len(px) - H):
        fwd[i] = statistics.pstdev(rets[i + 1:i + 1 + H]) * TD ** 0.5

    print(f"{'变量':12s} {'形态':10s} {'OOS MAE改善':>12s} {'预测vs真实r':>12s} {'方向命中':>9s}")
    print("-" * 62)
    rows = []
    for f in factors:
        p = os.path.join(MACRO, f"{f}.csv")
        if not os.path.exists(p):
            print(f"{f:12s} 缺数据")
            continue
        vals = align(dates, load_series(p))
        for form in ("水平", "20日变化"):
            xs = []
            for i in range(len(dates)):
                if form == "水平":
                    xs.append(vals[i])
                else:
                    xs.append(None if (vals[i] is None or vals[max(0, i - 20)] is None)
                              else vals[i] - vals[max(0, i - 20)])
            idx = [i for i in range(100, len(dates) - H)
                   if xs[i] is not None and cur[i] > 0 and fwd[i] is not None]
            if len(idx) < MIN_TRAIN + BLOCK:
                continue

            e1 = e2 = 0.0
            n = 0
            p2s, acts = [], []
            k = MIN_TRAIN
            while k < len(idx):
                tr = idx[:k]
                te = idx[k:k + BLOCK]
                X1 = [[1.0, math.log(cur[i])] for i in tr]
                X2 = [[1.0, math.log(cur[i]), xs[i]] for i in tr]
                y = [fwd[i] for i in tr]
                b1, b2 = ols(X1, y), ols(X2, y)
                if b1 and b2:
                    for i in te:
                        yhat1 = pred(b1, [1.0, math.log(cur[i])])
                        yhat2 = pred(b2, [1.0, math.log(cur[i]), xs[i]])
                        e1 += abs(fwd[i] - yhat1)
                        e2 += abs(fwd[i] - yhat2)
                        p2s.append(yhat2)
                        acts.append(fwd[i])
                        n += 1
                k += BLOCK
            if n < 100:
                continue
            mae1, mae2 = e1 / n, e2 / n
            gain = (mae1 - mae2) / mae1
            r = pearson(p2s, acts) if len(set(p2s)) > 1 else 0
            hit = sum(1 for a, b in zip(p2s, acts)
                      if (a > statistics.mean(p2s)) == (b > statistics.mean(acts))) / n
            rows.append((f, form, gain, r, hit, n))
            print(f"{f:12s} {form:10s} {gain*100:+11.1f}% {r:+12.3f} {hit*100:8.1f}%")

    if rows:
        rows.sort(key=lambda x: -x[2])
        print("\n=== 排名（按 OOS MAE 改善）===")
        for f, form, g, r, h, n in rows[:5]:
            verdict = "✅ 可用" if g > 0.05 else ("⚠️ 边缘" if g > 0.02 else "❌ 无增量")
            print(f"  {verdict} {f}/{form}: {g*100:+.1f}%  (r={r:+.3f}, n={n})")
        print("\n注意：即使 MAE 改善显著，也只说明『预测风险更准』，不说明能预测收益。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
