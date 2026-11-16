#!/usr/bin/env python3
"""fetch.py — 拉取真实日线数据（免 key），存到 ~/finance/data/*.csv

数据源（均已实测可用，2026-11-16）：
  A 股/ETF : 腾讯 https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
  加密     : Binance https://api.binance.com/api/v3/klines

用法：python3 fetch.py
"""
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

try:
    import akshare as ak
except Exception:
    ak = None

DATA = os.path.expanduser("~/finance/data")
TZ = timezone(timedelta(hours=8))

A_SHARE = {          # 腾讯代码 -> 文件名
    "sh000300": "csi300",     # 沪深300
    "sh000905": "csi500",     # 中证500
    "sh000016": "sse50",      # 上证50
    "sh000001": "sse_comp",   # 上证综指
    "sz399001": "szse_comp",  # 深证成指
    "sz399006": "chinext",    # 创业板指
    "sh000688": "star50",     # 科创50
    "sh518880": "gold",       # 黄金ETF
    "sh511260": "bond10y",    # 十年国债ETF
    "sh513100": "nasdaq_etf",  # 纳指ETF（QDII）
    "hkHSI": "hsi",           # 恒生指数
    "hkHSTECH": "hstech",     # 恒生科技
    "hkHSCEI": "hscei",       # 国企指数
    "sh000985": "csi_all",    # 中证全指
    "sh000852": "csi1000",    # 中证1000
    "sh000903": "csi100",     # 中证100
    "sh000932": "csi_consume",  # 中证消费
    "sz399673": "chinext50",  # 创业板50
    "sh513050": "china_internet",  # 中概互联ETF
    "sh512880": "sec_etf",    # 证券ETF
    "sh000012": "bond_index",  # 上证国债指数（6 年历史，补上债券缺口）
    "sh000922": "csi_div",    # 中证红利
}
CRYPTO = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol",
          "BNBUSDT": "bnb", "XRPUSDT": "xrp", "ADAUSDT": "ada",
          "DOGEUSDT": "doge", "LINKUSDT": "link", "AVAXUSDT": "avax",
          "DOTUSDT": "dot", "LTCUSDT": "ltc", "BCHUSDT": "bch",
          "TRXUSDT": "trx"}


def curl(url, timeout=25):
    """带重试与 WAF 检测：遇到 HTML 证明页（腾讯 WAF 501）就退避重试。"""
    last = ""
    for attempt in range(3):
        p = subprocess.run(["curl", "-sS", "--max-time", str(timeout), url],
                           capture_output=True, text=True)
        if p.returncode == 0:
            body = p.stdout.strip()
            if body and not body.startswith("<"):
                return body
            last = "返回 HTML（疑似 WAF 拦截）"
        else:
            last = p.stderr.strip()[:120]
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"连续 {3} 次失败：{last}")


def fetch_a(sym):
    """优先 akshare（新浪后端，24 年历史）→ 腾讯 → 新浪 JSON。"""
    if ak is not None:
        try:
            fn = ak.stock_zh_index_daily if sym.startswith(("sh00", "sz39")) else ak.stock_zh_a_daily
            kw = {"symbol": sym}
            if fn is ak.stock_zh_a_daily:
                kw["adjust"] = "qfq"
            df = fn(**kw)
            if df is not None and len(df) > 200:
                rows = [{"date": str(r["date"])[:10], "open": float(r["open"]),
                         "high": float(r["high"]), "low": float(r["low"]),
                         "close": float(r["close"]), "volume": float(r.get("volume") or 0)}
                        for _, r in df.iterrows()]
                return rows
        except Exception as e_ak:
            print(f"  [{sym}] akshare 失败({str(e_ak)[:50]})", file=sys.stderr)
    try:
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={sym},day,,,1500,qfq")
        d = json.loads(curl(url, timeout=15))
        node = d["data"][sym]
        bars = node.get("qfqday") or node.get("day") or []
        if not bars:
            raise RuntimeError("腾讯返回 0 根")
        return [{"date": b[0], "open": float(b[1]), "high": float(b[3]),
                 "low": float(b[4]), "close": float(b[2]),
                 "volume": float(b[5])} for b in bars]
    except Exception as e_tx:
        url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=1500")
        try:
            d = json.loads(curl(url, timeout=20))
            if not d:
                raise RuntimeError("新浪返回空")
            print(f"  [{sym}] 腾讯失败({str(e_tx)[:40]})，已切新浪", file=sys.stderr)
            return [{"date": b["day"], "open": float(b["open"]), "high": float(b["high"]),
                     "low": float(b["low"]), "close": float(b["close"]),
                     "volume": float(b.get("volume") or 0)} for b in d]
        except Exception as e_sn:
            raise RuntimeError(f"腾讯:{str(e_tx)[:40]} / 新浪:{str(e_sn)[:40]}")


def fetch_crypto(sym, pages=5):
    """分页向前取，最多 pages*1000 根（约 13 年）。"""
    all_rows = []
    end = None
    for _ in range(pages):
        url = (f"https://api.binance.com/api/v3/klines?symbol={sym}"
               f"&interval=1d&limit=1000" + (f"&endTime={end}" if end else ""))
        d = json.loads(curl(url))
        if not d:
            break
        chunk = []
        for k in d:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=TZ).strftime("%Y-%m-%d")
            chunk.append({"date": ts, "open": float(k[1]), "high": float(k[2]),
                          "low": float(k[3]), "close": float(k[4]),
                          "volume": float(k[5])})
        all_rows = chunk + all_rows
        end = d[0][0] - 1
        if len(d) < 1000:
            break
    # 去重排序
    seen = {}
    for r in all_rows:
        seen[r["date"]] = r
    return [seen[k] for k in sorted(seen)]


def save(name, rows):
    """按日期合并写盘：绝不因为某个降级源返回更少的数据而丢掉已有历史。"""
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{name}.csv")
    merged = {}
    if os.path.exists(path):
        try:
            for r in csv.DictReader(open(path)):
                d = r.get("date")
                if d:
                    merged[d] = {k: r[k] for k in
                                 ("date", "open", "high", "low", "close", "volume")}
        except Exception:
            merged = {}
    old_n = len(merged)
    for r in rows:
        merged[r["date"]] = {"date": r["date"],
                             "open": f"{r['open']:.4f}", "high": f"{r['high']:.4f}",
                             "low": f"{r['low']:.4f}", "close": f"{r['close']:.4f}",
                             "volume": f"{r['volume']:.0f}"}
    out = [merged[k] for k in sorted(merged)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(out)
    grew = "（合并）" if len(out) > len(rows) else ""
    print(f"{name:12s} {len(out):5d} bars  {out[0]['date']} → {out[-1]['date']}  "
          f"本次 {len(rows)} 根{grew}  {path}")


def main():
    ok = 0
    total = len(A_SHARE) + len(CRYPTO)
    for sym, name in A_SHARE.items():
        try:
            save(name, fetch_a(sym))
            ok += 1
        except Exception as e:
            print(f"{name}: FAIL {e}", file=sys.stderr)
        time.sleep(1.5)          # 腾讯 WAF 对突发请求很敏感
    for sym, name in CRYPTO.items():
        try:
            save(name, fetch_crypto(sym))
            ok += 1
        except Exception as e:
            print(f"{name}: FAIL {e}", file=sys.stderr)
        time.sleep(0.3)
    print(f"\n成功 {ok}/{total}")
    if ok == 0:
        print("全部失败——先确认是不是被 WAF 拦了（curl 看返回是否 HTML）", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
