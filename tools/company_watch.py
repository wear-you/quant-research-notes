#!/usr/bin/env python3
"""company_watch.py — 个股信息监控：公告、新闻、财务关键指标。

设计原则：
  1. 只做"事实归集 + 分级"，不做"利好/利空"的判断——那是市场的事
  2. 每条公告带日期与链接，可追溯
  3. 重大事件去重后写入 alerts.jsonl，供后续回看是否真的影响了价格

用法：
  python3 company_watch.py              # 监控 watchlist.txt 里的股票
  python3 company_watch.py 600519 000858
"""
import csv
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
import akshare as ak

FIN = os.path.expanduser("~/finance")
OUTDIR = os.path.join(FIN, "company")
WATCH = os.path.join(FIN, "watchlist.txt")
ALERTS = os.path.join(FIN, "company", "alerts.jsonl")
TZ = timezone(timedelta(hours=8))

# 重大程度分级（只按事实类型，不按好坏）
HIGH = ["业绩预告", "业绩快报", "季度报告", "年度报告", "半年度报告", "主要经营数据",
        "重大合同", "中标", "回购", "增持", "减持",
        "重组", "并购", "收购", "诉讼", "仲裁", "立案", "问询", "停牌",
        "退市", "分红", "股权激励", "非公开发行", "可转债", "质押", "冻结"]
MED = ["董事会", "股东大会", "投资者关系", "监事", "高管", "变更", "聘任"]


def load_watch():
    if len(sys.argv) > 1:
        return [a for a in sys.argv[1:] if a.isdigit()]
    if os.path.exists(WATCH):
        codes = [l.strip() for l in open(WATCH) if l.strip() and not l.startswith("#")]
        return [c for c in codes if c.isdigit()]
    return ["600519"]


def level_of(title):
    for k in HIGH:
        if k in title:
            return "高"
    for k in MED:
        if k in title:
            return "中"
    return "低"


def fetch_notices(code, days=45):
    end = datetime.now(TZ).date()
    start = end - timedelta(days=days)
    try:
        df = ak.stock_zh_a_disclosure_report_cninfo(
            symbol=code, market="沪深京",
            start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        out = []
        for _, r in df.iterrows():
            out.append({"code": code, "date": str(r["公告时间"])[:10],
                        "title": str(r["公告标题"]).strip(),
                        "url": str(r.get("公告链接", ""))})
        return out
    except Exception as e:
        print(f"  [{code}] 公告失败 {type(e).__name__}")
        return []


def fetch_news(code, n=5):
    try:
        df = ak.stock_news_em(symbol=code)
        return [{"date": str(r["发布时间"])[:16], "title": str(r["新闻标题"]).strip(),
                 "src": str(r.get("文章来源", ""))}
                for _, r in df.head(n).iterrows()]
    except Exception:
        return []


def fetch_fin(code):
    """取最近四期的核心指标。"""
    try:
        df = ak.stock_financial_abstract(symbol=code)
    except Exception:
        return {}
    want = {"归母净利润": ["归母净利润", "归属于母公司所有者的净利润"],
            "营业总收入": ["营业总收入", "营业收入"],
            "净资产收益率": ["净资产收益率", "加权净资产收益率"],
            "每股收益": ["基本每股收益", "每股收益"]}
    periods = [c for c in df.columns if str(c)[:2].isdigit()][:4]
    out = {}
    for label, keys in want.items():
        for _, r in df.iterrows():
            if any(k in str(r["指标"]) for k in keys):
                out[label] = {p: r[p] for p in periods}
                break
    return {"periods": periods, "data": out}


def prev_alerts():
    if not os.path.exists(ALERTS):
        return set()
    out = set()
    for line in open(ALERTS):
        try:
            d = json.loads(line)
            out.add((d["code"], d["date"], d["title"]))
        except Exception:
            continue
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    codes = load_watch()
    seen = prev_alerts()
    new_alerts = []
    L = [f"# 公司监控 · {today}", "", f"> 监控 {len(codes)} 只：{', '.join(codes)}", ""]

    for code in codes:
        notices = fetch_notices(code)
        news = fetch_news(code)
        fin = fetch_fin(code)

        L.append(f"## {code}")
        L.append("")
        if fin.get("data"):
            p = fin["periods"]
            L.append(f"**财务（{p[0] if p else ''}）**")
            L.append("")
            L.append("| 指标 | " + " | ".join(str(x) for x in p) + " |")
            L.append("|---" * (len(p) + 1) + "|")
            for k, v in fin["data"].items():
                L.append(f"| {k} | " + " | ".join(str(v.get(x, "-")) for x in p) + " |")
            L.append("")

        high = [n for n in notices if level_of(n["title"]) == "高"]
        L.append(f"**近 45 天公告 {len(notices)} 条，其中高关注 {len(high)} 条**")
        L.append("")
        for n in notices[:12]:
            lv = level_of(n["title"])
            mark = {"高": "🔴", "中": "🟡", "低": "⚪"}[lv]
            L.append(f"- {mark} {n['date']} {n['title']}")
            if lv == "高":
                key = (code, n["date"], n["title"])
                if key not in seen:
                    new_alerts.append({"code": code, "date": n["date"],
                                       "title": n["title"], "level": lv,
                                       "url": n.get("url", ""),
                                       "logged_at": today})
        L.append("")
        if news:
            L.append("**近期新闻**")
            L.append("")
            for x in news[:3]:
                L.append(f"- {x['date']} {x['title']}（{x['src']}）")
            L.append("")

    with open(ALERTS, "a") as f:
        for a in new_alerts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    path = os.path.join(OUTDIR, f"{today}.md")
    open(path, "w").write("\n".join(L))
    print("\n".join(L[:30]))
    print(f"\n新增高关注事件 {len(new_alerts)} 条")
    print(f"报告: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
