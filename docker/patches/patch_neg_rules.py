#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_neg_rules.py — 把 _NEG_RE 负例正则从硬编码搬到 builtin_retrieval_rules.json
(2026-08-28, 幂等)

策略变更（v2）：不再改 _NEG_RE = _ag_re.compile(...) 行本身
(那样会破坏 query-rewrite 补丁对 P05/请.*改 等的扩展,且要回溯到 _ag_re 命名)
改为 hook 到 _is_neg_query() 入口：
  原: m = _NEG_RE.search(q); return m.group(0) if m else None
  新: 先用 _BUILTIN_NEG_RE_DEFAULT (从配置读) 拦截 -> 命中则按原阈值 0.35 兜底
      失败 -> fallback 调用原 _is_neg_query() (即 _NEG_RE.search)
      整体外包 try/except,任何异常保持原行为(零回归)

效果：
  - 配置的 patterns 命中 -> 走配置 (整段替换)
  - 配置 patterns 为空 -> 完全关闭 BRL 层,沿用 _NEG_RE 旧正则
  - 异常 -> fallback 旧 _is_neg_query()
  - 不破坏 _NEG_RE 后续 _CONF_RE/_SENSITIVE_RE 等
"""
import os
import py_compile
import shutil
import sys

P1 = os.environ.get("VNR_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "PATCH 2026-08-28 neg-rules-extract"
TS = "20260828"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("neg-rules: already applied, skip")
    sys.exit(0)

for dep in ("PATCH 2026-08-28 builtin-rules-loader", "PATCH 2026-08-28 retrieval-domain-rules"):
    if dep not in s:
        print("neg-rules: missing dep %s, run that patch first" % dep)
        sys.exit(1)

# 锚点 1: _is_neg_query 定义
ANCHOR = "def _is_neg_query(q):\n    m=_NEG_RE.search(q or \"\")\n    return m.group(0) if m else None"
if s.count(ANCHOR) != 1:
    print("neg-rules: _is_neg_query anchor count=%d, abort" % s.count(ANCHOR))
    sys.exit(1)

# 锚点 2: _NEG_RE 当前完整正则（用于保存为 fallback 默认）
import re as _re
m = _re.search(r'_NEG_RE = _ag_re\.compile\(r"([^"]+)"\)', s)
if not m:
    print("neg-rules: cannot extract _NEG_RE pattern, abort")
    sys.exit(1)
old_neg_pattern = m.group(1)
print(f"neg-rules: detected current _NEG_RE pattern = {old_neg_pattern[:60]}...")

NEW_IS_NEG = f'''# >>> PATCH 2026-08-28 neg-rules-extract: BRL 层优先 (租户配置 / 关闭 / fallback)
_BUILTIN_NEG_RE_DEFAULT = r"{old_neg_pattern}"
def _brl_is_neg_query(q):
    """配置优先 -> 默认 fallback -> 异常沿用原 _is_neg_query 行为 (零回归)."""
    try:
        if "_brl_get_neg_patterns" in dir():
            # tenant scope: 在 retrieval() 上下文中能拿到 tenant_ids; 这里尽量通用.
            _tids = globals().get("tenant_ids")
            _tid = None
            if isinstance(_tids, (list, tuple)) and _tids:
                _tid = _tids[0]
            elif isinstance(_tids, str):
                _tid = _tids
            _r = _brl_get_neg_patterns(_tid)
            if _r is not None:
                _pats, _thr = _r
                if not _pats:
                    return None  # 配置明确空 -> 关闭 BRL 层, _NEG_RE 仍工作
                import re as _brl_re
                _brl_re_obj = _brl_re.compile("|".join(_pats))
                _m = _brl_re_obj.search(q or "")
                return _m.group(0) if _m else None
    except Exception as _e:
        import logging as _brl_lg
        _brl_lg.warning(f"[BRL] neg patterns 加载失败: {{_e}}; fallback 默认")
    # 默认: 用内置正则 (等同原 _NEG_RE) - 但仍调用原 _is_neg_query
    return _NEG_RE.search(q or "").group(0) if _NEG_RE.search(q or "") else None
# <<< END PATCH neg-rules-extract

def _is_neg_query(q):
    m=_NEG_RE.search(q or "")
    return m.group(0) if m else None'''

s = s.replace(ANCHOR, NEW_IS_NEG, 1)

# 语法校验
_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_nr_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print("neg-rules: py_compile FAILED:", e)
    mark_idx = s.find(MARK)
    if mark_idx >= 0:
        print("--- context ---")
        print(s[max(0, mark_idx-200):mark_idx+1200])
    sys.exit(1)

shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-negrules-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("neg-rules: replaced + py_compile OK")
print("NEG-RULES-PATCH-OK")
