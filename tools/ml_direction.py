#!/usr/bin/env python3
"""ml_direction.py — 用「走线图 + 公司消息」预测涨跌方向。

设计要点：
  1. 只预测方向（涨/跌），不预测幅度——这是用户要求的，也是最容易验证的
  2. 严格时间切分：只用过去训练，预测未来；绝不随机打乱
  3. **对照组**：纯价格特征 vs 价格+消息特征。差多少，就是消息值的钱
  4. 预测目标同时看原始收益与超额收益（剔除沪深300）

特征：
  价格（走线图）：5/20/60 日收益、均线偏离、20 日波动、回撤、量比
  消息（公告）：近 5/20 日各类公告计数——业绩预增/预减、回购、增减持、
              诉讼、停牌、重组、分红、定期报告、全部公告

模型：L2 正则逻辑回归（numpy 实现，无 sklearn）
"""
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np

warnings.filterwarnings("ignore")
import akshare as ak

FIN = os.path.expanduser("~/finance")
CACHE = os.path.join(FIN, "cache")
PX_CACHE = os.path.join(CACHE, "px.json")
ANN_CACHE = os.path.join(CACHE, "ann.json")
TZ = timezone(timedelta(hours=8))

N_STOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
START = "2023-01-01"

CATS = {
    "yjyg_pos": ["业绩预增", "预增", "业绩预告"],
    "yjyg_neg": ["预减", "首亏", "续亏"],
    "buyback": ["回购"],
    "reduce": ["减持"],
    "increase": ["增持"],
    "litigation": ["诉讼", "仲裁", "立案", "处罚"],
    "suspend": ["停牌", "终止上市", "退市"],
    "restructure": ["重组", "并购", "收购", "重大资产"],
    "dividend": ["分红", "利润分配"],
    "report": ["季度报告", "年度报告", "半年度报告", "业绩快报"],
}


def load_json(p, default):
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return default


def save_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"))


def universe(n):
    df = ak.index_stock_cons_csindex(symbol="000300")
    codes = [str(c).zfill(6) for c in df["成分券代码"].tolist()]
    return codes[:n]


def prices(code, cache):
    if code in cache and cache[code]:
        return cache[code]
    sym = f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}"
    try:
        df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
        if df is not None and len(df) > 100:
            cache[code] = [[str(r["date"])[:10], float(r["close"]),
                            float(r.get("volume", 0) or 0)] for _, r in df.iterrows()]
            return cache[code]
    except Exception:
        pass
    return []


def announcements(code, cache):
    if code in cache:
        return cache[code]
    end = datetime.now(TZ).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京",
            start_date=START.replace("-", ""), end_date=end)
        rows = [[str(r["公告时间"])[:10], str(r["公告标题"])] for _, r in df.iterrows()]
    except Exception:
        rows = []
    cache[code] = rows
    return rows


def flags(title):
    out = []
    for k, kws in CATS.items():
        if any(w in title for w in kws):
            out.append(k)
    return out


def build():
    px_cache = load_json(PX_CACHE, {})
    ann_cache = load_json(ANN_CACHE, {})
    codes = universe(N_STOCKS)
    print(f"股票池 {len(codes)} 只，目标 {H} 日方向")

    bench = {}
    for r in __import__("csv").DictReader(open(os.path.join(FIN, "data", "csi300.csv"))):
        bench[r["date"]] = float(r["close"])

    rows = []
    for i, c in enumerate(codes):
        px = prices(c, px_cache)
        if len(px) < 200:
            continue
        ann = announcements(c, ann_cache)
        # 按日期索引公告
        byd = {}
        for d, t in ann:
            byd.setdefault(d, []).append(t)
        dates = [x[0] for x in px]
        closes = np.array([x[1] for x in px])
        vols = np.array([x[2] for x in px])
        rets = np.diff(closes) / closes[:-1]
        rets = np.insert(rets, 0, 0.0)

        for k in range(70, len(px) - H):
            d = dates[k]
            if d < START:
                continue
            ma20 = closes[k - 19:k + 1].mean()
            ma60 = closes[k - 59:k + 1].mean()
            vol20 = rets[k - 19:k + 1].std()
            vol60 = rets[k - 59:k + 1].std()
            dd60 = closes[k] / closes[k - 59:k + 1].max() - 1
            v5 = vols[k - 4:k + 1].mean()
            v20 = vols[k - 19:k + 1].mean()
            vratio = v5 / v20 if v20 > 0 else 1.0

            # 公告计数
            cnt = {f"e_{k2}_5": 0 for k2 in CATS}
            cnt.update({f"e_{k2}_20": 0 for k2 in CATS})
            all5 = all20 = 0
            for j in range(max(0, k - 20), k + 1):
                dj = dates[j]
                if dj not in byd:
                    continue
                all20 += len(byd[dj])
                if j >= k - 4:
                    all5 += len(byd[dj])
                for t in byd[dj]:
                    for f in flags(t):
                        cnt[f"e_{f}_20"] += 1
                        if j >= k - 4:
                            cnt[f"e_{f}_5"] += 1

            fwd = closes[k + H] / closes[k] - 1
            d_end = dates[k + H]
            b0, b1 = bench.get(d), bench.get(d_end)
            exc = fwd - (b1 / b0 - 1) if (b0 and b1) else None

            rows.append({
                "date": d,
                "y": 1 if fwd > 0 else 0,
                "y_exc": (1 if exc > 0 else 0) if exc is not None else None,
                "fwd": fwd, "exc": exc,
                **{f"p_{n}": v for n, v in {
                    "r5": closes[k] / closes[k - 5] - 1,
                    "r20": closes[k] / closes[k - 20] - 1,
                    "r60": closes[k] / closes[k - 60] - 1,
                    "ma20d": closes[k] / ma20 - 1,
                    "ma60d": closes[k] / ma60 - 1,
                    "vol20": vol20, "vol60": vol60,
                    "dd60": dd60, "vratio": vratio,
                }.items()},
                "all5": all5, "all20": all20,
                **cnt,
            })
        save_json(PX_CACHE, px_cache)
        save_json(ANN_CACHE, ann_cache)
        print(f"  [{i+1}/{len(codes)}] {c} 累计 {len(rows)} 行")

    save_json(PX_CACHE, px_cache)
    save_json(ANN_CACHE, ann_cache)
    return rows


PRICE_FEATS = ["p_r5", "p_r20", "p_r60", "p_ma20d", "p_ma60d",
               "p_vol20", "p_vol60", "p_dd60", "p_vratio"]
EVENT_FEATS = [f"e_{k}_5" for k in CATS] + [f"e_{k}_20" for k in CATS] + ["all5", "all20"]


def fit_logit(X, y, l2=1.0, iters=400, lr=0.5):
    n, k = X.shape
    w = np.zeros(k)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = X.T @ (p - y) / n + l2 * w / n
        gb = (p - y).mean()
        w -= lr * g
        b -= lr * gb
    return w, b


def predict(X, w, b):
    return 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))


def evaluate(y, p, label):
    base = y.mean()
    acc = ((p > 0.5).astype(int) == y).mean()
    brier = ((p - y) ** 2).mean()
    brier_base = ((base - y) ** 2).mean()
    skill = 1 - brier / brier_base if brier_base > 0 else 0
    # AUC
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5
    print(f"  {label:22s} 准确率 {acc*100:5.2f}%  AUC {auc:.3f}  "
          f"Brier技巧 {skill:+.4f}  (基准涨率 {base*100:.1f}%)")
    return {"acc": acc, "auc": auc, "skill": skill}


def main():
    rows = build()
    if len(rows) < 2000:
        print("样本不足")
        return 1
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"\n总样本 {len(df)}  区间 {df['date'].min()} → {df['date'].max()}")

    dates = sorted(df["date"].unique())
    cuts = [dates[int(len(dates) * f)] for f in (0.6, 0.8)]
    print(f"切分点 {cuts[0]} / {cuts[1]}（只用过去训练）\n")

    for target in ("y", "y_exc"):
        sub = df[df[target].notna()].copy()
        if len(sub) < 1000:
            continue
        sub[target] = sub[target].astype(int)
        print(f"=== 预测目标：{'涨跌' if target=='y' else '超额涨跌'} ===")
        for feats, name in ((PRICE_FEATS, "纯走线图"),
                            (PRICE_FEATS + EVENT_FEATS, "走线图+公司消息")):
            allp, ally = [], []
            for c in cuts:
                tr = sub[sub["date"] < c]
                te = sub[(sub["date"] >= c)]
                if len(tr) < 500 or len(te) < 200:
                    continue
                mu = tr[feats].mean()
                sd = tr[feats].std().replace(0, 1)
                Xtr = ((tr[feats] - mu) / sd).values
                Xte = ((te[feats] - mu) / sd).values
                w, b = fit_logit(Xtr, tr[target].values.astype(float))
                allp.append(predict(Xte, w, b))
                ally.append(te[target].values)
            if allp:
                evaluate(np.concatenate(ally), np.concatenate(allp), name)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
