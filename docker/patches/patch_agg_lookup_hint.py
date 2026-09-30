#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_agg_lookup_hint.py — 优化聚合分类 (2026-08-28, 幂等)

目的: 把 7 个 miss 中"含精确编号的题"从聚合路径中摘出,让纯检索来回答。
逻辑: 在 _ag_light_classify 内部最前面加前置判断
      题面同时含 强精确编号 (PRC-\w+ | 物料编码)  且  不含聚合信号词(最高/最低/低于/最多/统计...)
      → 直接 return False 跳过聚合
"""
import os
import py_compile
import shutil
import sys

P1 = os.environ.get("VAL_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "PATCH 2026-08-28 agg-lookup-hint"
TS = "20260828"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("agg-lookup-hint: already applied, skip")
    sys.exit(0)

# 锚点: def 行本身
ANCHOR = "def _ag_light_classify(q):"
if s.count(ANCHOR) != 1:
    print("agg-lookup-hint: anchor count=%d, abort" % s.count(ANCHOR))
    sys.exit(1)

NEW_INNER = ANCHOR + '''
    # >>> PATCH 2026-08-28 agg-lookup-hint: 精确编号 + 单一具体查询 -> 走纯检索
    if _ag_re.search(r"(?i)物料编码\\s*[A-Z]{2,5}-?\\w+|PRC-\\w+|prc-\\w+|物料\\s*编码\\s*[A-Z]\\w+-\\d+", q or ""):
        if not _ag_re.search(r"最高|最低|低于|最多|最少|众数|统计|合计|总计|全部|所有|多于|不少于", q or ""):
            print(f"[AGG-LOOKUP-HINT] skip-agg q={(q or '')[:32]}", flush=True)
            return False
    # <<< END PATCH agg-lookup-hint'''

s = s.replace(ANCHOR, NEW_INNER, 1)

_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_al_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print("agg-lookup-hint: py_compile FAILED:", e)
    import re as _re
    m = _re.search(MARK, s)
    if m:
        print(s[max(0, m.start()-200):m.start()+800])
    sys.exit(1)

shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-agglh-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("agg-lookup-hint: applied + py_compile OK")
print("AGG-LOOKUP-HINT-PATCH-OK")
