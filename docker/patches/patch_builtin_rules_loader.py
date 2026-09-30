#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_builtin_rules_loader.py — 通用加载函数注入补丁 (2026-08-28，幂等)

目的：把 _NEG_RE / _domain_rules / LLM PROMPT 模板中的硬编码集中到
    /ragflow/conf/builtin_retrieval_rules.json，可被 env 一键关闭。
设计：
  - env `RAGFLOW_BUILTIN_RULES=0` -> 全部段返回空 dict/空 list（纯基线，他人租户用）
  - env `RAGFLOW_BUILTIN_RULES_PATH=/path/...json` -> 自定义路径（默认 /ragflow/conf/builtin_retrieval_rules.json）
  - 配置 mtime 30s 热加载（无文件系统事件 -> 时间戳轮询，轻量）
  - 解析失败/缺失 -> search.py 内部常量 fallback（爱博当前行为零回归）
  - tenant scope：配置含 `_apply_tenant_ids` 非空 -> 仅列表内租户加载；其他租户直接 fallback

注入位置：search.py 模块级，HELPER_ANCHOR = "# <<< END PATCH llm-route-v2 helper"
（与 llm-route-v2 helper 同位置追加，互不冲突）
"""
import os
import py_compile
import shutil
import sys

P1 = os.environ.get("VBL_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "PATCH 2026-08-28 builtin-rules-loader"
TS = "20260828"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("loader: already applied, skip")
    sys.exit(0)

HELPER_ANCHOR = "# <<< END PATCH llm-route-v2 helper"
if s.count(HELPER_ANCHOR) != 1:
    print("loader: helper anchor count=%d, abort" % s.count(HELPER_ANCHOR))
    sys.exit(1)

# 注入到 helper 锚点之后(模块级,无缩进)
LOADER = '''

# >>> PATCH 2026-08-28 builtin-rules-loader: 把 _NEG_RE / _domain_rules / LLM PROMPT 模板
#     从硬编码搬到 /ragflow/conf/builtin_retrieval_rules.json。env RAGFLOW_BUILTIN_RULES=0 全关闭，
#     配置 mtime 30s 热加载，tenant scope + 解析失败 fallback。零回归。
import os as _brl_os
import time as _brl_time
import json as _brl_json
_BRL_ENV_DISABLE = _brl_os.environ.get("RAGFLOW_BUILTIN_RULES", "1") == "0"
_BRL_CFG_PATH = _brl_os.environ.get("RAGFLOW_BUILTIN_RULES_PATH", "/ragflow/conf/builtin_retrieval_rules.json")
_BRL_STATE = {"cfg": None, "mtime": 0.0, "ts": 0.0}
_BRL_HOT_SEC = 30.0

def _brl_load_cfg():
    """30s 热加载配置。返回 dict(可能为空)。任何异常返回空 dict,调用方 fallback 到内置默认."""
    if _BRL_ENV_DISABLE:
        return {}
    st = _BRL_STATE
    now = _brl_time.time()
    if st["cfg"] is not None and (now - st["ts"]) < _BRL_HOT_SEC:
        return st["cfg"]
    cfg = {}
    try:
        if _brl_os.path.exists(_BRL_CFG_PATH):
            mt = _brl_os.path.getmtime(_BRL_CFG_PATH)
            if st["cfg"] is None or mt != st["mtime"]:
                with open(_BRL_CFG_PATH, "r", encoding="utf-8") as _f:
                    cfg = _brl_json.load(_f)
                st["mtime"] = mt
    except Exception as _e:
        import logging as _brl_lg
        _brl_lg.warning(f"[BRL] 配置加载失败({_BRL_CFG_PATH}): {type(_e).__name__}: {_e}; 返回空")
        cfg = {}
    st["cfg"] = cfg if isinstance(cfg, dict) else {}
    st["ts"] = now
    return st["cfg"]

def _brl_tenant_allowed(cfg, tenant_id):
    """tenant scope 判定: _apply_tenant_ids 缺失/空 -> 全部放行; 非空 -> 必须命中列表."""
    if not isinstance(cfg, dict):
        return True
    scope = cfg.get("_apply_tenant_ids")
    if scope is None or scope == [] or scope == "":
        return True
    if not tenant_id:
        return False
    return str(tenant_id) in [str(x) for x in scope]

def _brl_get_domain_rules(tenant_id=None):
    """返回 [(pattern_str, [doc_keys]), ...]. 优先级: 配置 -> 内置默认(list 形式) -> [].
    配置项格式: {"pattern": "...", "doc_keys": [...]} 序列; 内置默认(爱博行为)放在调用方."""
    cfg = _brl_load_cfg()
    if not _brl_tenant_allowed(cfg, tenant_id):
        return None  # 调用方据此判定走默认
    dr = cfg.get("domain_rules")
    if not dr:
        return None
    out = []
    for item in dr:
        if not isinstance(item, dict):
            continue
        p = item.get("pattern")
        dk = item.get("doc_keys") or []
        if p and isinstance(dk, list):
            out.append((p, [str(x) for x in dk]))
    return out

def _brl_get_neg_patterns(tenant_id=None):
    """返回 (patterns_list, threshold). patterns 是 regex str 列表,调用方用 re.compile('|'.join(...))."""
    cfg = _brl_load_cfg()
    if not _brl_tenant_allowed(cfg, tenant_id):
        return None
    np = cfg.get("neg_patterns")
    if not isinstance(np, dict):
        return None
    pats = np.get("patterns") or []
    thr = np.get("neg_threshold")
    if not pats:
        return None
    return [str(x) for x in pats], float(thr) if thr is not None else 0.35

def _brl_get_llm_route_examples(tenant_id=None):
    """返回 llm_route_examples dict; 调用方按需 .get(...)."""
    cfg = _brl_load_cfg()
    if not _brl_tenant_allowed(cfg, tenant_id):
        return None
    ex = cfg.get("llm_route_examples")
    return ex if isinstance(ex, dict) else None
# <<< END PATCH builtin-rules-loader
'''

# 注入到 HELPER_ANCHOR 之后(锚点前是 # 注释行,留空行隔开)
s = s.replace(HELPER_ANCHOR, HELPER_ANCHOR + "\n" + LOADER, 1)

# 语法校验 + 备份
_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_brl_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
py_compile.compile(tmp, doraise=True)
shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-brlloader-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("loader: injected + py_compile OK")
print("BRL-LOADER-PATCH-OK")
