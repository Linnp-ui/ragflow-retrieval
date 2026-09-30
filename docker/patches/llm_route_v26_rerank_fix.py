#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm-route-v2.6 (2026-08-26)：count/latest/年份类条件 BAAI rerank（对齐客户端 L0 策略）。

问题：服务端路由 _lr_v2_scoped 只用内部 rerank_with_knn（term/dense fusion），永远不走
rerank_id（BAAI）通道；而 count/latest/年份类恰是标准 L0 路径必传 rerank_id 的题类
（报告实测：计数/最新类无 rerank 时 rank 深，rerank 后 rank1 sim 0.987+）。
基线 sres 是否已过 BAAI 取决于请求是否带 rerank_id（数据集 search_config），服务端 chat/MCP
不按题类条件启用，故这类题在路由路径下丢 rerank。

修复：路由合并后，对题类命中的请求用租户默认 RERANK 模型对合并候选集（baseline∪scoped）
统一重排，覆盖两类情形：
  - picked 非空（count/complex 等有 scoped 域内召回）：重排 baseline+scoped 合并集
  - picked 为 None（最新/年份被 gate 直通、无 scoped）：仅重排 baseline
软回退：题类不匹配 / 租户无 RERANK 模型 / 异常 -> 返回 None，保持原序（零回归）。
幂等（MARK 守卫），先 py_compile 校验新内容再写盘，备份在 /tmp。

注：本补丁直接改挂载唯一事实源 docker/patches/search.py（host 路径），容器内 /ragflow/rag/nlp/search.py
为其 bind mount；改后 docker restart 重载即可，无需重 build 镜像。
"""
import os
import py_compile
import shutil
import sys

# 挂载唯一事实源（host 路径；容器内对应 /ragflow/rag/nlp/search.py）
# 本地校验可用 V26_TARGET 覆盖（如 Windows 绝对路径）；生产默认即 host 挂载路径。
P1 = os.environ.get("V26_TARGET", "/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
MARK = "llm-route-v2.6"
TS = "20260826"

s = open(P1, encoding="utf-8").read()
if MARK in s:
    print("v2.6 already applied, skip")
    sys.exit(0)

# ---------------------------------------------------------------------------
# 1) helper 注入：锚定 v2 helper 结束标记，追加 _lr_v2_rerank_combined（模块级）
# ---------------------------------------------------------------------------
HELPER_ANCHOR = "# <<< END PATCH llm-route-v2 helper"
if s.count(HELPER_ANCHOR) != 1:
    print("helper anchor count=%d, abort" % s.count(HELPER_ANCHOR))
    sys.exit(1)

HELPER2 = '''

async def _lr_v2_rerank_combined(self, question, tenant_id, sres, term_w, vec_w, rank_feature):
    """count/latest/年份类条件 BAAI rerank：对齐客户端 L0 策略，对合并候选集（baseline∪scoped）统一重排。
    无 rerank 模型 / 题类不匹配 / 异常 -> 返回 None（保持原序）。"""
    import re as _rr
    import asyncio as _ra
    q = question or ""
    # 与客户端 L0 rerank 适用类一致：计数/汇总 + 最新/版本 + 年份；无反斜杠转义避免注入转义问题
    _RERANK_CLASS = _rr.compile(
        r"多少|几台|几次|几个|几份|几处|几套|合计|总共|总计|汇总|总数|数量|用量|最多|最少|共计|共[0-9]|"
        r"最新|当前|最近|最新版|版本|(?:19|20)[0-9]{2}[ ]*年|[0-9]{8}|[0-9]{4}[-/.][0-9]{1,2}([-/.][0-9]{1,2})?")
    if not _RERANK_CLASS.search(q):
        return None
    if not getattr(sres, "ids", None):
        return None
    try:
        from common.constants import LLMType
        from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
        from api.db.services.llm_service import LLMBundle
        mcfg = await _ra.to_thread(get_tenant_default_model_by_type, tenant_id, LLMType.RERANK)
        if not mcfg:
            return None
        rmdl = LLMBundle(tenant_id, mcfg)
        try:
            sim, tsim, vsim = self.rerank_by_model(rmdl, sres, q, term_w, vec_w, rank_feature=rank_feature)
        finally:
            try:
                rmdl.close()
            except Exception:
                pass
        return (sim, tsim, vsim)
    except Exception as _e:
        logging.warning(f"[LLM-ROUTE-v2.6] rerank_by_model 失败: {type(_e).__name__}: {_e}")
        return None
'''
s = s.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPER2, 1)

# ---------------------------------------------------------------------------
# 2) 调用点注入：锚定 else 的 skip 日志行；在其后（与 if/else 同级的 20 空格缩进）插入
#    rerank 块，使其对 picked 与非 picked 两种情形都执行。
# ---------------------------------------------------------------------------
CALL_ANCHOR = '                        logging.info(f"[LLM-ROUTE-v2] skip q={question[:24]} why={_lr_why}")'
if s.count(CALL_ANCHOR) != 1:
    print("call anchor count=%d, abort" % s.count(CALL_ANCHOR))
    sys.exit(1)

CALL2 = '''                        logging.info(f"[LLM-ROUTE-v2] skip q={question[:24]} why={_lr_why}")
                    # >>> PATCH 2026-08-26 llm-route-v2.6: count/latest/年份类条件 BAAI rerank（对齐客户端 L0；覆盖 picked 与非 picked）
                    try:
                        _lr_rk = await _lr_v2_rerank_combined(self, question, _lr_tids[0], sres, term_similarity_weight, vector_similarity_weight, rank_feature)
                        if _lr_rk is not None:
                            sim_np = np.array(_lr_rk[0], dtype=np.float64)
                            tsim = list(_lr_rk[1]); vsim = list(_lr_rk[2])
                            valid_idx = [i for i in valid_idx if float(sim_np[i]) >= post_threshold]
                            valid_idx.sort(key=lambda i: -float(sim_np[i]))
                            filtered_count = len(valid_idx)
                            ranks["total"] = int(filtered_count)
                            print(f"[LLM-ROUTE-v2.6] reranked q={question[:24]} valid={filtered_count}", flush=True)
                    except Exception as _lr_re:
                        logging.warning(f"[LLM-ROUTE-v2.6] rerank 失败保持原序: {type(_lr_re).__name__}: {_lr_re}")
                    # <<< END PATCH llm-route-v2.6'''
s = s.replace(CALL_ANCHOR, CALL2, 1)

# ---------------------------------------------------------------------------
# 语法校验：先编译新内容（临时文件），通过后再写盘 + 备份，避免破坏 live 文件
# ---------------------------------------------------------------------------
_tmpdir = os.path.dirname(P1)
tmp = os.path.join(_tmpdir, "search_v26_check_%s.py" % TS)
open(tmp, "w", encoding="utf-8").write(s)
py_compile.compile(tmp, doraise=True)

shutil.copy(P1, os.path.join(_tmpdir, "search.py.bak-llmroute-v26-%s" % TS))
open(P1, "w", encoding="utf-8").write(s)
print("search.py: v2.6 rerank patch applied + py_compile OK")
print("LLM-ROUTE-V2.6-PATCH-OK")
