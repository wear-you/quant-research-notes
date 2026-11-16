#!/usr/bin/env python3
"""probe_macro.py — 探测宏观/流动性/情绪类数据在无账户条件下的可得性。

只回答一个问题：哪些源真的能拉到数据、拉到多少行、更新到哪天。
不给结论，先看弹药库。
"""
import warnings

warnings.filterwarnings("ignore")

import akshare as ak

PROBES = [
    ("CPI 月度", lambda: ak.macro_china_cpi_monthly()),
    ("PPI 月度", lambda: ak.macro_china_ppi()),
    ("PMI 月度", lambda: ak.macro_china_pmi()),
    ("货币供应 M0/M1/M2", lambda: ak.macro_china_money_supply()),
    ("社融规模", lambda: ak.macro_china_shrzgm()),
    ("中美国债收益率", lambda: ak.bond_zh_us_rate()),
    ("Shibor", lambda: ak.macro_china_shibor_all()),
    ("LPR 报价", lambda: ak.macro_china_lpr()),
    ("融资融券(沪)", lambda: ak.stock_margin_sse()),
    ("融资融券(深)", lambda: ak.stock_margin_szse()),
    ("北向资金", lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="北上")),
    ("指数估值(沪深300)", lambda: ak.stock_zh_index_value_csindex(symbol="000300")),
    ("人民币汇率", lambda: ak.currency_boc_sina(symbol="美元", start_date="20200101", end_date="20261231")),
    ("原油期货", lambda: ak.futures_foreign_hist(symbol="CL")),
    ("VIX 恐慌指数", lambda: ak.index_vix()),
    ("A股成交额", lambda: ak.stock_zh_a_hist(symbol="000001", period="daily",
                                             start_date="20200101", end_date="20261231")),
]

for name, fn in PROBES:
    try:
        df = fn()
        if df is None or len(df) == 0:
            print(f"❌ {name:18s} 空")
            continue
        cols = list(df.columns)[:5]
        print(f"✅ {name:18s} {len(df):6d} 行  列: {cols}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:70]
        print(f"❌ {name:18s} {type(e).__name__}: {msg}")
