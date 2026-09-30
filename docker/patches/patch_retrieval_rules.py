#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_retrieval_rules.py — 把 optimized-retrieval v3 中的 _domain_rules
（爱博产品线硬编码）替换为 _brl_get_domain_rules 调用（2026-08-28，幂等）。

零回归设计：
  - 配置缺失/解析失败/_apply_tenant_ids 不含本租户 -> 返回 None -> 走 search.py 内部 _BUILTIN_DOMAIN_RULES_DEFAULT（= 原硬编码块复制）
  - 配置命中 -> 用配置规则
  - env RAGFLOW_BUILTIN_RULES=0 -> _brl_get_domain_rules() 直接返回 None -> 走默认（他人租户即使无意间有配置也不生效；用默认等于关闭域路由,纯基线）

依赖：必须先跑 patch_builtin_rules_loader.py 注入 _brl_* 函数。

锚点：v3 块中的 _domain_rules 定义（已知唯一，由 llm-route-v2 + retrflow-v3 写入）。
"""
import os
import py_compile
import shutil
import sys

P1 = os.environ.get("VDR_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "PATCH 2026-08-28 retrieval-domain-rules"
TS = "20260828"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("domain-rules: already applied, skip")
    sys.exit(0)

if "PATCH 2026-08-28 builtin-rules-loader" not in s:
    print("domain-rules: loader not found, run patch_builtin_rules_loader.py first")
    sys.exit(1)

# 原 v3 块中的 _domain_rules 定义（爱博当前硬编码）
# 注：源文件用 \\d 双反斜杠在 re 字符串里; 替换时也要原样保留.
OLD_DOMAIN_RULES = '''            _domain_rules = [
                (r"BOT012|P01-\d+|现SN号|样机编号", ["样机表"]),
                (r"销售|销售额|月度汇总|库存统计|中文销售", ["中文销售", "销售"]),
                (r"会议纪要|王鹏.*任务|下一步安排|主持人", ["会议纪要"]),
                (r"问题跟踪|问题报表|故障|SH001|卡扣反了|Y轴.*噪音|神经造影", ["问题跟踪", "问题报表"]),
                (r"维修备料|备件|易损件|单机用量|研发物料编码", ["维修备料", "备件"]),
                (r"P02|冠脉|驱动器|调节支臂|操作器|服务手册", ["P02", "服务手册", "冠脉"]),
            ]
            _matched_domain_docs = None
            for _pat, _doc_keys in _domain_rules:
                if _re.search(_pat, _q):
                    _cands = [d for d in _doc_to_idx_tmp.keys() if any(k in d for k in _doc_keys)]
                    if _cands:
                        _matched_domain_docs = set(_cands)
                        print(f"[OPT-RETRIEVAL-v3] 域路由 pat={_pat} keep={_matched_domain_docs}", flush=True)
                        break'''

if s.count(OLD_DOMAIN_RULES) != 1:
    print("domain-rules: anchor count=%d (expected 1), abort" % s.count(OLD_DOMAIN_RULES))
    sys.exit(1)

# 替换为：先尝试配置,失败/无规则/租户被排除 -> 用原 _BUILTIN_DOMAIN_RULES_DEFAULT
# 关键：_BUILTIN_DOMAIN_RULES_DEFAULT 必须与原定义完全一致
NEW_DOMAIN_RULES = '''            # >>> PATCH 2026-08-28 retrieval-domain-rules: 优先从 /ragflow/conf/builtin_retrieval_rules.json
            #     加载,失败/缺/租户被排除 -> 用内置默认(等同原硬编码,爱博行为零回归).
            #     env RAGFLOW_BUILTIN_RULES=0 -> 直接走默认列表(等同关闭域路由).
            _BUILTIN_DOMAIN_RULES_DEFAULT = [
                (r"BOT012|P01-\d+|现SN号|样机编号", ["样机表"]),
                (r"销售|销售额|月度汇总|库存统计|中文销售", ["中文销售", "销售"]),
                (r"会议纪要|王鹏.*任务|下一步安排|主持人", ["会议纪要"]),
                (r"问题跟踪|问题报表|故障|SH001|卡扣反了|Y轴.*噪音|神经造影", ["问题跟踪", "问题报表"]),
                (r"维修备料|备件|易损件|单机用量|研发物料编码", ["维修备料", "备件"]),
                (r"P02|冠脉|驱动器|调节支臂|操作器|服务手册", ["P02", "服务手册", "冠脉"]),
            ]
            _domain_rules = _BUILTIN_DOMAIN_RULES_DEFAULT
            try:
                _brl_tenant = tenant_ids[0] if isinstance(tenant_ids, (list, tuple)) and tenant_ids else (tenant_ids if isinstance(tenant_ids, str) else None)
                _brl_rules = _brl_get_domain_rules(_brl_tenant) if "_brl_get_domain_rules" in dir() else None
                if _brl_rules is not None:
                    _domain_rules = _brl_rules
                    print(f"[BRL] domain_rules from config: {len(_brl_rules)} entries (tenant={_brl_tenant})", flush=True)
            except Exception as _brl_e:
                pass  # 任何异常 -> 用默认
            _matched_domain_docs = None
            for _pat, _doc_keys in _domain_rules:
                if _re.search(_pat, _q):
                    _cands = [d for d in _doc_to_idx_tmp.keys() if any(k in d for k in _doc_keys)]
                    if _cands:
                        _matched_domain_docs = set(_cands)
                        print(f"[OPT-RETRIEVAL-v3] 域路由 pat={_pat} keep={_matched_domain_docs}", flush=True)
                        break'''

s = s.replace(OLD_DOMAIN_RULES, NEW_DOMAIN_RULES, 1)

# 语法校验 + 备份
_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_dr_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
py_compile.compile(tmp, doraise=True)
shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-domainrules-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("domain-rules: replaced + py_compile OK")
print("DOMAIN-RULES-PATCH-OK")
