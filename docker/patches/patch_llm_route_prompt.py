#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_llm_route_prompt.py — 把 llm-route-v2 PROMPT 里的 P0x 例子
(P01样机表__20240410更新、P01样机表__20251215更新) 替换为可配置模板 (2026-08-28, 幂等)

PROMPT 改动：
  原：硬编码
    "多版本快照文档选择规则：\n"
    "- 同名前缀的文档系列（如 P01样机表__20240410更新、P01样机表__20251215更新）是同一台账的不同版本快照，\n"
    "- 问具体实体（编号/序列号/人名/设备）的存放、状态、用途，无法判断实体在哪个版本时：同时选出该系列的全部版本快照文档（不要只选日期最大的）\n"
    "- 问题明确含\"最新/当前\"时，只选该系列日期最大的版本\n"
    "- 问某具体年份数据时选对应该年份的文档\n"

  新：根据 _brl_get_llm_route_examples(tenant) 动态拼：
    - versioned_doc_example  -> 例子(默认 P0x 例子,可改为 device_table_2024Q1 等通用名)
    - versioned_doc_phrase  -> "同一台账的不同版本快照"(可改)
    - 其余规则句保持(本来就是通用规则)
  当配置缺失/租户被排除/env 关闭 -> 走内置默认 = 原文(爱博行为零回归)

依赖：必须先跑 patch_builtin_rules_loader.py。
锚点：v2 helper 中的 _LLM_ROUTE_V2_PROMPT = (...) 长字符串。
"""
import os
import py_compile
import shutil
import sys

P1 = os.environ.get("VLP_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "PATCH 2026-08-28 llm-route-prompt"
TS = "20260828"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("llm-prompt: already applied, skip")
    sys.exit(0)

if "PATCH 2026-08-28 builtin-rules-loader" not in s:
    print("llm-prompt: loader not found, run patch_builtin_rules_loader.py first")
    sys.exit(1)

# 锚点: _LLM_ROUTE_V2_PROMPT = ( ... ) 块。
# 用特征字面量定位: 'P01样机表__20240410更新' (已替换则跳过此步,直接走 .format 替换)
NEEDLE = "P01样机表__20240410更新"
if NEEDLE in s:
    # 只替换 1 行: "如 P01样机表__20240410更新、P01样机表__20251215更新"
    OLD_EXAMPLE = "P01样机表__20240410更新、P01样机表__20251215更新"
    if s.count(OLD_EXAMPLE) != 1:
        print("llm-prompt: old example count=%d (expected 1), abort" % s.count(OLD_EXAMPLE))
        sys.exit(1)
    # 替换为格式化字符串: {example} 占位符,运行时按 tenant 读
    NEW_EXAMPLE = "{_brl_versioned_doc_example}"
    s = s.replace(OLD_EXAMPLE, NEW_EXAMPLE, 1)
    print("llm-prompt: P0x example replaced")
else:
    print("llm-prompt: P0x example already replaced, skip")

# 找到 NEEDLE 所在的多行字符串(PROMPT). 重写整个 PROMPT 字面量.
# 简单策略: 用一个 sentinel 字符串作为锚点 start, 下一个 ')' + '\n' 作为 end.
# 但更安全是直接替换 NEEDLE 上下文段.

# 我们替换从"多版本快照文档选择规则："到下一行"...选对应该年份的文档"的内容
# 用 re.DOTALL 多行替换,直接处理源字符串.

# 由于源文件实际缩进未知(可能含 \\n 转义),我们用两层策略:
# 1) 找到 PROMPT 起始 '多版本快照文档选择规则' 之前的 \n (这个块在一行 r"..." 里)
# 2) 找到 '选对应该年份的文档' 之后的 \"
# 3) 替换中间段

# 实际 PROMPT 在源里长这样(从 llm_route_v2_patch.py 看出):
#   _LLM_ROUTE_V2_PROMPT = (
#     "你是知识库文档路由器...。\\n"
#     "多版本快照文档选择规则:\\n"
#     "- 同名前缀的文档系列（如 P01样机表__20240410更新、P01样机表__20251215更新）是同一台账的不同版本快照，每个版本只含当时的行，行内容可能只在某几个版本中出现\\n"
#     "- 问具体实体（编号/序列号/人名/设备）的存放、状态、用途，无法判断实体在哪个版本时：同时选出该系列的全部版本快照文档（不要只选日期最大的）\\n"
#     "- 问题明确含\\"最新/当前\\"时，只选该系列日期最大的版本\\n"
#     "- 问某具体年份数据时选对应该年份的文档\\n"
#     "文档清单（名称 — 一句话摘要）：\\n{docs}\\n"
#     ...
# 我们用 PROMPT 变量名做锚点重写整段. 但更稳的是只改一个具体子串.

# PROMPT 引用处: 把 _LLM_ROUTE_V2_PROMPT.format(docs=..., q=...) 改成:
#   先按 tenant 拿到 example,再 format.
# 锚点: _LLM_ROUTE_V2_PROMPT.format(docs=...
PROMPT_FMT = "_LLM_ROUTE_V2_PROMPT.format(docs="
if s.count(PROMPT_FMT) != 1:
    print("llm-prompt: _LLM_ROUTE_V2_PROMPT.format anchor count=%d, abort" % s.count(PROMPT_FMT))
    sys.exit(1)

OLD_FMT_LINE = "    prompt = " + PROMPT_FMT + r'"\n".join(lines), q=question)' + "\n"
if s.count(OLD_FMT_LINE) != 1:
    print("llm-prompt: OLD_FMT_LINE count=%d (expected 1), abort" % s.count(OLD_FMT_LINE))
    sys.exit(1)
# 单行替换: prompt = (lambda ...) ( ... )
NEW_FMT_BLOCK = '''# >>> PATCH 2026-08-28 llm-route-prompt: 把 _LLM_ROUTE_V2_PROMPT.format 中的硬编码 P0x 例子
#     替换为可配置模板变量 {_brl_versioned_doc_example}; 运行时按 tenant 读 builtin_retrieval_rules.json
    prompt = (
                lambda _ex: _LLM_ROUTE_V2_PROMPT.format(docs="\\n".join(lines), q=question, _brl_versioned_doc_example=_ex)
            )(
                _brl_get_llm_route_examples(
                    (tenant_ids[0] if isinstance(tenant_ids, (list, tuple)) and tenant_ids else tenant_ids)
                    if isinstance(tenant_ids, (str, type(None))) or
                       (isinstance(tenant_ids, (list, tuple)) and (not tenant_ids or isinstance(tenant_ids[0], str)))
                    else None
                ).get("versioned_doc_example", "P01样机表__20240410更新、P01样机表__20251215更新")
                if "_brl_get_llm_route_examples" in dir() else
                "P01样机表__20240410更新、P01样机表__20251215更新"
            )
'''

s = s.replace(OLD_FMT_LINE, NEW_FMT_BLOCK, 1)

# 语法校验
_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_lp_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
try:
    py_compile.compile(tmp, doraise=True)
except py_compile.PyCompileError as e:
    print("llm-prompt: py_compile FAILED:")
    print(e)
    mark_idx = s.find(MARK)
    if mark_idx < 0:
        # 找 PROMPT_FMT 替换后的位置
        mark_idx = s.find('_brl_get_llm_route_examples')
    if mark_idx >= 0:
        print("--- context around replacement ---")
        print(s[max(0, mark_idx-200):mark_idx+800])
    sys.exit(1)

shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-llmprompt-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("llm-prompt: replaced + py_compile OK")
print("LLM-PROMPT-PATCH-OK")
