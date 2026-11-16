#!/usr/bin/env python3
"""factors.py — 参数地图的实测版：哪些外部变量真的含有信息？

背景：
  已验证：收益方向预测失败（技巧分数为负），波动预测部分成功（沪深300 R² 0.33）。
  下一步不是找更多"预测收益"的指标，而是找**能改善风险判断**的变量，
  以及验证宏观/流动性变量是否与未来收益有稳定关系（预期：很弱）。

做法：
  1. 拉取无账户可得的宏观/流动性/情绪序列
  2. 对齐到沪深300 交易日（前向填充）
  3. 对每个变量的三种形态（水平、20日变化、1年 z 分数）做：
     - 与未来 20 日波动的相关性
     - 与未来 20 日收益的相关性
     - 在"当前波动"之上能否改善波动预测（样本外 MAE）
  4. 排名

注意：这是**筛选**，不是验证。相关性显著的变量必须再做严格因果回放才能用。
"""
import csv
import math
import os
import statistics
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
import akshare as ak

DATA = os.path.expanduser("~/finance/data")
MACRO = os.path.join(DATA, "macro")
TD = 244
H = 20


def ensure_dir():
    os.makedirs(MACRO, exist_ok=True)


# ---------- 日期解析 ----------
def parse_any(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y年%m月"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    # '2026年10月份'
    import re
    m = re.search(r"(\d{4})\D+(\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01"
    return None


def to_series(df, date_col, val_col, invert=False):
    out = {}
    for _, r in df.iterrows():
        d = parse_any(r[date_col])
        if d is None:
            continue
        try:
            v = float(r[val_col])
        except (TypeError, ValueError):
            continue
        if v != v:  # NaN
            continue
        out[d] = v
    items = sorted(out.items())
    return items


def save(name, items):
    p = os.path.join(MACRO, f"{name}.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "value"])
        w.writerows(items)
    print(f"  {name:16s} {len(items):6d} 点  {items[0][0]} → {items[-1][0]}")


def fetch_all():
    ensure_dir()
    print("=== 拉取 ===")
    out = {}

    def add(name, fn):
        try:
            items = fn()
            if len(items) > 50:
                save(name, items)
                out[name] = items
            else:
                print(f"  {name:16s} 样本不足({len(items)})")
        except Exception as e:
            print(f"  {name:16s} FAIL {type(e).__name__}: {str(e)[:50]}")

    def yld(col):
        def f():
            df = ak.bond_zh_us_rate()
            return to_series(df, "日期", col)
        return f

    add("cn10y", yld("中国国债收益率10年"))
    add("cn2y", yld("中国国债收益率2年"))
    add("cn_slope", yld("中国国债收益率10年-2年"))
    add("us10y", yld("美国国债收益率10年"))

    add("shibor_on", lambda: to_series(ak.macro_china_shibor_all(), "日期", "O/N-定价"))
    add("shibor_3m", lambda: to_series(ak.macro_china_shibor_all(), "日期", "3M-定价"))

    def lpr():
        df = ak.macro_china_lpr()
        return to_series(df, "TRADE_DATE", "LPR1Y")
    add("lpr1y", lpr)

    add("margin", lambda: sorted(to_series(ak.stock_margin_sse(), "信用交易日期", "融资余额").items()))

    def fx():
        df = ak.currency_boc_sina(symbol="美元", start_date="20150101", end_date="20261231")
        return to_series(df, "日期", "央行中间价")
    add("usdcny", fx)

    def oil():
        df = ak.futures_foreign_hist(symbol="CL")
        return to_series(df, "date", "close")
    add("oil", oil)

    add("cpi", lambda: to_series(ak.macro_china_cpi_monthly(), "日期", "今值"))
    add("ppi", lambda: to_series(ak.macro_china_ppi(), "月份", "当月同比增长"))
    add("pmi", lambda: to_series(ak.macro_china_pmi(), "月份", "制造业-指数"))
    add("m2", lambda: to_series(ak.macro_china_money_supply(), "月份", "货币和准货币(M2)-同比增长"))
    return out


# ---------- 对齐 ----------
def load_csi():
    p = os.path.join(DATA, "csi300.csv")
    rows = [(r["date"], float(r["close"])) for r in csv.DictReader(open(p))]
    return rows


def align(dates, series):
    """把 series 前向填充到 dates 上。"""
    out, j, last = [], 0, None
    for d in dates:
        while j < len(series) and series[j][0] <= d:
            last = series[j][1]
            j += 1
        out.append(last)
    return out


def feature_forms(vals, i, i20, iy):
    """返回 (水平, 20日变化, 1年z分数)。"""
    lvl = vals[i]
    chg = vals[i] - vals[i20] if vals[i20] is not None else None
    hist = [v for v in vals[max(0, i - 244):i + 1] if v is not None]
    z = None
    if len(hist) > 60:
        m, sd = statistics.mean(hist), statistics.pstdev(hist)
        if sd > 0 and lvl is not None:
            z = (lvl - m) / sd
    return lvl, chg, z


def corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 100:
        return None, 0
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs) / n)
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in pairs) / n)
    if sx == 0 or sy == 0:
        return None, n
    cov = sum((p[0] - mx) * (p[1] - my) for p in pairs) / n
    return cov / (sx * sy), n


def ols(X, y):
    """X: list of rows (含常数项)；返回系数。高斯消元。"""
    k = len(X[0])
    A = [[sum(X[n][i] * X[n][j] for n in range(len(X))) for j in range(k)] + [sum(X[n][i] * y[n] for n in range(len(X)))]
         for i in range(k)]
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


def predict(beta, row):
    return sum(b * x for b, x in zip(beta, row))


def main():
    data = fetch_all()
    csi = load_csi()
    dates = [d for d, _ in csi]
    px = [p for _, p in csi]

    rets = [0.0] + [px[i] / px[i - 1] - 1 for i in range(1, len(px))]
    cur_vol = [statistics.pstdev(rets[max(0, i - 60):i + 1]) * TD ** 0.5
               for i in range(len(px))]
    fwd_vol = [None] * len(px)
    fwd_ret = [None] * len(px)
    for i in range(len(px) - H):
        fwd_vol[i] = statistics.pstdev(rets[i + 1:i + 1 + H]) * TD ** 0.5
        fwd_ret[i] = px[i + H] / px[i] - 1

    aligned = {k: align(dates, v) for k, v in data.items()}

    results = []
    for name, vals in aligned.items():
        lvl, chg, z = [], [], []
        for i in range(len(dates)):
            a, b, c = feature_forms(vals, i, max(0, i - 20), None)
            lvl.append(a)
            chg.append(b)
            z.append(c)
        for form, arr in (("水平", lvl), ("20日变化", chg), ("1年z", z)):
            r_vol, n1 = corr(arr, fwd_vol)
            r_ret, n2 = corr(arr, fwd_ret)
            if r_vol is None or n1 < 200:
                continue
            # 增量：log(cur_vol) → +factor，样本外 MAE
            cut = int(len(dates) * 0.7)
            X1, X2, yv, idx = [], [], [], []
            for i in range(100, len(dates) - H):
                if arr[i] is None or cur_vol[i] <= 0 or fwd_vol[i] is None:
                    continue
                X1.append([1.0, math.log(cur_vol[i])])
                X2.append([1.0, math.log(cur_vol[i]), arr[i]])
                yv.append(fwd_vol[i])
                idx.append(i)
            if len(yv) < 300:
                continue
            tr = [k for k, i in enumerate(idx) if i < cut]
            te = [k for k, i in enumerate(idx) if i >= cut]
            if len(tr) < 200 or len(te) < 100:
                continue
            b1 = ols([X1[k] for k in tr], [yv[k] for k in tr])
            b2 = ols([X2[k] for k in tr], [yv[k] for k in tr])
            if not b1 or not b2:
                continue
            mae1 = sum(abs(yv[k] - predict(b1, X1[k])) for k in te) / len(te)
            mae2 = sum(abs(yv[k] - predict(b2, X2[k])) for k in te) / len(te)
            gain = (mae1 - mae2) / mae1
            results.append({"var": name, "form": form, "r_vol": r_vol,
                            "r_ret": r_ret, "n": n1, "gain": gain})

    results.sort(key=lambda r: -abs(r["r_vol"]))
    print("\n=== 与未来 20 日波动 / 收益的相关性（全样本）===")
    print(f"{'变量':14s} {'形态':8s} {'r(波动)':>8s} {'r(收益)':>8s} {'样本外MAE改善':>12s}")
    for r in results:
        print(f"{r['var']:14s} {r['form']:8s} {r['r_vol']:+8.3f} {r['r_ret']:+8.3f} "
              f"{r['gain']*100:+11.1f}%")

    print("\n=== 值得进一步验证（|r(波动)|>0.15 或 MAE 改善>2%）===")
    cand = [r for r in results if abs(r["r_vol"]) > 0.15 or r["gain"] > 0.02]
    if not cand:
        print("  无")
    for r in cand:
        print(f"  {r['var']} / {r['form']}: r(波动)={r['r_vol']:+.3f}  "
              f"r(收益)={r['r_ret']:+.3f}  MAE改善 {r['gain']*100:+.1f}%")
    print("\n注意：全样本相关 + 单次切分只是筛选。入选变量必须再跑严格因果回放。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
