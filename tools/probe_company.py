#!/usr/bin/env python3
"""probe_company.py — 探测个股信息类数据的可得性。

关注四类：
  1. 公告（交易所披露）
  2. 业绩预告 / 业绩快报（最可能驱动股价的定期事件）
  3. 个股新闻
  4. 财务摘要
"""
import warnings

warnings.filterwarnings("ignore")
import akshare as ak

SYM = "600519"

PROBES = [
    ("全市场公告(按日)", lambda: ak.stock_notice_report(symbol="全部", date="20261114")),
    ("业绩预告", lambda: ak.stock_yjyg_em(date="20260930")),
    ("业绩快报", lambda: ak.stock_yjkb_em(date="20260930")),
    ("个股公告(巨潮)", lambda: ak.stock_zh_a_disclosure_report_cninfo(
        symbol=SYM, market="沪深京", start_date="20260101", end_date="20261116")),
    ("个股新闻", lambda: ak.stock_news_em(symbol=SYM)),
    ("财务摘要", lambda: ak.stock_financial_abstract(symbol=SYM)),
    ("财务分析指标", lambda: ak.stock_financial_analysis_indicator(symbol=SYM, start_year="2023")),
    ("龙虎榜", lambda: ak.stock_lhb_detail_em(start_date="20261110", end_date="20261116")),
    ("机构调研", lambda: ak.stock_jgdy_tj_em(date="20261001")),
]

for name, fn in PROBES:
    try:
        df = fn()
        if df is None or len(df) == 0:
            print(f"❌ {name:18s} 空")
            continue
        print(f"✅ {name:18s} {len(df):6d} 行  列: {list(df.columns)[:6]}")
    except Exception as e:
        print(f"❌ {name:18s} {type(e).__name__}: {str(e)[:60]}")
