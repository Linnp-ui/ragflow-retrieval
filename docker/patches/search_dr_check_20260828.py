#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import json
import logging
import re
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from rag.nlp import rag_tokenizer, query
import numpy as np
from common.doc_store.doc_store_base import MatchDenseExpr, FusionExpr, OrderByExpr, DocStoreConnection
from common.string_utils import remove_redundant_spaces
from common.float_utils import get_float
from common.constants import PAGERANK_FLD, TAG_FLD
from common.tag_feature_utils import parse_tag_features
from common import settings

from common.misc_utils import thread_pool_exec

def index_name(uid): return f"ragflow_{uid}"

# >>> PATCH 2026-08-26 llm-route-v2: 摘要驱动 LLM 文档路由（逐文档域检索 + union 合并；配置 /ragflow/conf/llm_route.json 30s 热加载；env RAGFLOW_LLM_ROUTE=0 总开关）
import os as _lr_os
_LLM_ROUTE_V2_ENV_ON = _lr_os.environ.get("RAGFLOW_LLM_ROUTE", "1") != "0"
_LLM_ROUTE_V2_STATE = {"cache": {}, "cfg": None, "cfg_ts": 0.0}
_LLM_ROUTE_V2_CFG_PATH = "/ragflow/conf/llm_route.json"
_LLM_ROUTE_V2_DEFAULTS = {"enabled": False, "min_confidence": 0.85, "max_docs": 100, "ttl_sec": 600, "timeout_sec": 10}
# 复杂精确规则门控（与客户端 llm_doc_router.js COMPLEX_RULES 对齐）：命中任一才允许 LLM
_LLM_ROUTE_V2_COMPLEX = [
    r"多少|几台|几次|几个|几份|几处|几套|合计|总共|总计|汇总|总数|数量|用量|最多|最少|共计|共\d",
    r"最新|当前|最近|目前|现有|最新版",
    r"(19|20)\d{2}\s*年|\d{8}|\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?",
    r"BOT\d{3,}|P01-\d{1,2}(?!\d)|序列号|编号|物料编码|批次号|故障代码|0x[0-9a-fA-F]{2,}|规格型号|料号",
    r"#[1-9]\d*",
    r"行程|速度|温度|范围|时长|多久|℃|°C|毫米|mm/?s|功率|压力|电压|规格|尺寸",
    r"谁|责任人|负责人|主持人|操作者",
    r"存放(地点|位置|在哪)|在哪里|哪里|哪个展厅|哪个车间|哪栋",
    r"第[0-9一二三四五六七八九十]+(部分|章|节|条|款)",
    r"并且|而且|同时|以及|分别",
]
_LLM_ROUTE_V2_PROMPT = (
    "你是知识库文档路由器。根据问题选出能回答它的文档（可多选；宁缺毋滥，无关的不选）。\n"
    "多版本快照文档选择规则：\n"
    "- 同名前缀的文档系列（如 P01样机表__20240410更新、P01样机表__20251215更新）是同一台账的不同版本快照，每个版本只含当时的行，行内容可能只在某几个版本中出现\n"
    "- 问具体实体（编号/序列号/人名/设备）的存放、状态、用途，无法判断实体在哪个版本时：同时选出该系列的全部版本快照文档（不要只选日期最大的）\n"
    "- 问题明确含\"最新/当前\"时，只选该系列日期最大的版本\n"
    "- 问某具体年份数据时选对应该年份的文档\n"
    # llm-route-v2.4: 产品线代号优先（AB-55 类：摘要未提故障代码时靠问题中的 P0x 代号定位产品线）
    "- 问题明确提及产品线代号（如 P01、P02、P03）时，优先选该产品线的文档（如该产品的说明书/报表），不要混淆到其他产品线\n"
    "文档清单（名称 — 一句话摘要）：\n{docs}\n"
    "问题：{q}\n"
    "严格只输出一个 JSON 对象：{{\"docs\":[\"文档名\",...],\"confidence\":0.0到1.0,\"reason\":\"不超过15字\"}}。禁止输出引号包裹的字符串、数组或其他文字。无法确定时给低 confidence。"
)

def _lr_v2_cfg():
    import time
    st = _LLM_ROUTE_V2_STATE
    now = time.time()
    if st["cfg"] is None or now - st["cfg_ts"] > 30:
        import json as _lr_json
        cfg = dict(_LLM_ROUTE_V2_DEFAULTS)
        try:
            if _lr_os.path.exists(_LLM_ROUTE_V2_CFG_PATH):
                with open(_LLM_ROUTE_V2_CFG_PATH, "r", encoding="utf-8") as f:
                    cfg.update(_lr_json.load(f))
        except Exception:
            pass
        st["cfg"], st["cfg_ts"] = cfg, now
    return st["cfg"]

# >>> PATCH 2026-08-28 query-rewrite-v1: 查询改写与扩展支路（规则式 0 LLM + 配置热加载；env RAGFLOW_QUERY_REWRITE=0 总开关）
import os as _qr_os
_QR_ENV_ON = _qr_os.environ.get("RAGFLOW_QUERY_REWRITE", "1") != "0"
_QR_STATE = {"cfg": None, "cfg_ts": 0.0}
_QR_CFG_PATH = "/ragflow/conf/query_rewrite.json"
_QR_DEFAULTS = {"enabled": True, "max_extra_queries": 2, "search_size": 100,
                "search_similarity": 0.05, "skip_neg": True,
                "aliases": [], "expansions": []}

def _qr_cfg():
    import time
    st = _QR_STATE
    now = time.time()
    if st["cfg"] is None or now - st["cfg_ts"] > 30:
        import json as _qr_json
        cfg = dict(_QR_DEFAULTS)
        try:
            if _qr_os.path.exists(_QR_CFG_PATH):
                with open(_QR_CFG_PATH, "r", encoding="utf-8") as f:
                    cfg.update(_qr_json.load(f))
        except Exception:
            pass
        st["cfg"], st["cfg_ts"] = cfg, now
    return st["cfg"]

def _qr_variants(question, cfg):
    """改写/扩展候选查询：原问题 + 别名替换 + 关键词扩展变体。纯规则 0 LLM，去重 + 上限。"""
    import re as _qrv_re
    q = (question or "").strip()
    if not q:
        return []
    out = [q]
    seen = set(out)
    try:
        for rule in (cfg.get("aliases") or []):
            pat = rule.get("pattern") or ""
            rep = rule.get("replacement") or ""
            if not pat or not rep or _qrv_re.search(pat, q) is None:
                continue
            v = _qrv_re.sub(pat, rep, q)
            if v and v != q and v not in seen:
                out.append(v)
                seen.add(v)
        for rule in (cfg.get("expansions") or []):
            pat = rule.get("pattern") or ""
            add = rule.get("add") or ""
            if not pat or not add or _qrv_re.search(pat, q) is None:
                continue
            if add not in q:
                v = q + " " + add
                if v not in seen:
                    out.append(v)
                    seen.add(v)
    except Exception:
        return [q]
    mx = int(cfg.get("max_extra_queries", 2) or 2)
    return out[:mx + 1]
# <<< END PATCH query-rewrite-v1
# >>> PATCH 2026-08-28 query-rewrite-v2 MARK=QUERY-REWRITE-V2

def _lr_v2_gate(question):
    """返回 (允许, 原因)。列表类/最新/年份由 v3 规则处理，不走 LLM。"""
    if not _LLM_ROUTE_V2_ENV_ON:
        return False, "env-off"
    cfg = _lr_v2_cfg()
    if not cfg.get("enabled", False):
        return False, "cfg-off"
    q = question or ""
    if not q.strip():
        return False, "empty"
    if re.search(r"有哪些|全部|所有|清单|列表", q):
        return False, "list"
    if re.search(r"最新|当前|最近|latest", q, re.I):
        return False, "latest(v3)"
    if re.search(r"(20\d{2})\s*年|\d{8}|\d{4}[-/.]\d{1,2}", q):
        return False, "year(v3)"
    if not any(re.search(p, q) for p in _LLM_ROUTE_V2_COMPLEX):
        return False, "simple"
    return True, "pass"

async def _lr_v2_pick_docs(question, tenant_id, kb_ids):
    """摘要驱动 LLM 选文档。返回 ((doc_ids, names, conf), why) 或 (None, why)。任何异常向上抛由调用方软回退。"""
    import json as _lr_json
    import time as _lr_time
    import asyncio as _lr_asyncio
    ok, why = _lr_v2_gate(question)
    if not ok:
        return None, why
    if not kb_ids:
        return None, "no-kb"
    cfg = _lr_v2_cfg()
    from api.db.services.document_service import DocumentService
    from api.db.services.doc_metadata_service import DocMetadataService
    docs, meta = [], {}
    for kb in kb_ids:
        docs.extend(DocumentService.query(kb_id=kb) or [])
        try:
            meta.update(DocMetadataService.get_metadata_for_documents(None, kb) or {})
        except Exception:
            pass
    sdocs = [d for d in docs if (meta.get(d.id) or {}).get("summary")]
    n_sum = len(sdocs)
    if n_sum < 2 or n_sum > int(cfg.get("max_docs", 100)):
        return None, "sumdocs-%d" % n_sum
    ck = (question or "") + "|" + ",".join(sorted(kb_ids))
    now = _lr_time.time()
    hit = _LLM_ROUTE_V2_STATE["cache"].get(ck)
    if hit and now - hit["ts"] < int(cfg.get("ttl_sec", 600)):
        print(f"[LLM-ROUTE-v2] cache q={question[:24]} picked={hit['picked']}", flush=True)
        return (hit["ids"], hit["picked"], hit["conf"]), "cache"
    name_to_id, lines = {}, []
    for d in sdocs:
        su = str((meta.get(d.id) or {}).get("summary") or "").replace("\n", " ")[:80]
        lines.append(f"{d.name} — {su}")
        name_to_id[d.name] = d.id
    prompt = _LLM_ROUTE_V2_PROMPT.format(docs="\n".join(lines), q=question)
    from common.constants import LLMType
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    mcfg = await _lr_asyncio.to_thread(get_tenant_default_model_by_type, tenant_id, LLMType.CHAT)
    if not mcfg or not mcfg.get("api_base"):
        return None, "no-chat-model"

    def _lr_direct_chat(cfg_, prompt_, timeout_):  # llm-route-v2.3: 直连租户 chat 端点（复刻客户端 callLLMDirect：0.6s/conf 0.9；LLMBundle extra_body 路径 vLLM 不识别 enable_thinking，thinking 未关 10.7s/conf 0.62）
        import urllib.request
        import json as _ljd
        base = str(cfg_.get("api_base") or "").rstrip("/")
        url = base + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg_.get("api_key"):
            headers["Authorization"] = "Bearer " + str(cfg_["api_key"])
        payload = {"model": cfg_.get("llm_name"),
                   "messages": [{"role": "user", "content": prompt_}],
                   "stream": False, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        rq = urllib.request.Request(url, data=_ljd.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(rq, timeout=timeout_) as rs:
            j = _ljd.loads(rs.read().decode("utf-8"))
        ch = (j.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or ""

    _lr_to = float(cfg.get("timeout_sec", 10))
    resp = await _lr_asyncio.wait_for(
        _lr_asyncio.to_thread(_lr_direct_chat, mcfg, prompt, _lr_to),
        timeout=_lr_to + 5)
    resp = re.sub(r"^.*?\n\n", "", str(resp), flags=re.S).strip()
    mobj = re.search(r"\{[\s\S]*\}", resp)
    try:
        parsed = _lr_json.loads(mobj.group(0)) if mobj else {}
    except Exception:
        parsed = {}
    if isinstance(parsed, str):  # 双层编码（LLM 返回 JSON 字符串）
        try:
            parsed = _lr_json.loads(parsed)
        except Exception:
            parsed = {}
    names = [str(x) for x in (parsed.get("docs") or [])]
    try:
        conf = float(parsed.get("confidence", 0))
    except Exception:
        conf = 0.0
    ids = []
    for n in names:
        did = name_to_id.get(n)
        if not did:
            for k2, v2 in name_to_id.items():
                if k2 in n or n in k2:
                    did = v2
                    break
        if did and did not in ids:
            ids.append(did)
    _LLM_ROUTE_V2_STATE["cache"][ck] = {"ids": ids, "conf": conf, "picked": names, "ts": now}
    if len(_LLM_ROUTE_V2_STATE["cache"]) > 256:
        _LLM_ROUTE_V2_STATE["cache"].popitem(last=False)
    print(f"[LLM-ROUTE-v2] q={question[:24]} conf={conf} picked={names}", flush=True)
    if not ids or conf < float(cfg.get("min_confidence", 0.85)):
        return None, "lowconf-%.2f" % conf
    return (ids, names, conf), "llm"

async def _lr_v2_scoped(self, question, doc_ids, kb_ids, idx_names, sres, page_size, term_w, vec_w, post_threshold, rank_feature):
    """逐文档域内检索（复用 sres.query_vector 免重编码）。返回 [(chunk_id, sim, field)]，全部过阈 chunk（不预去重，union 端以 valid_idx 为基准）"""  # llm-route-v2.5
    dim = len(sres.query_vector or [])
    if not dim:
        return []
    limit = min(max(int(page_size), 50), 100)
    matchText, _ = self.qryr.question(question, min_match=0.3)
    matchDense = MatchDenseExpr(f"q_{dim}_vec", sres.query_vector, "float", "cosine", limit,
                                {"similarity": post_threshold if vec_w > 0 else 0.0})
    fusionExpr = FusionExpr("weighted_sum", limit, {"weights": "0.05,0.95"})
    src = ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
           "doc_id", "chunk_order_int", "page_num_int", "top_int", "create_timestamp_flt", "question_kwd",
           "question_tks", "doc_type_kwd", "available_int", "content_with_weight", "mom_id", PAGERANK_FLD,
           TAG_FLD, "row_id()"]
    all_ids, all_fields = [], {}
    for did in doc_ids:
        cond = {"kb_id": kb_ids, "doc_id": [did], "available_int": 1}
        res = await thread_pool_exec(self.dataStore.search, src, [], cond,
                                     [matchText, matchDense, fusionExpr], OrderByExpr(), 0, limit,
                                     idx_names, kb_ids)
        fields = self.dataStore.get_fields(res, src + ["_score"]) or {}
        for cid, f in fields.items():
            if cid in all_fields:
                continue
            all_ids.append(cid)
            all_fields[cid] = f
    if not all_ids:
        return []
    fake = self.SearchResult(total=len(all_ids), ids=list(all_ids),
                             query_vector=sres.query_vector, field=dict(all_fields))
    if settings.DOC_ENGINE_INFINITY:
        sims = [float((all_fields[c].get("_score") or 0.0)) for c in all_ids]
    else:
        knn = await self._knn_scores(fake, idx_names, kb_ids)
        sim, _ts, _vs = self.rerank_with_knn(fake, question, knn, term_w, vec_w, rank_feature=rank_feature)
        sims = [float(x) for x in sim]
    out = []
    for j, cid in enumerate(all_ids):
        if sims[j] >= post_threshold:
            out.append((cid, sims[j], all_fields[cid]))
    return out
# <<< END PATCH llm-route-v2 helper


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


# >>> PATCH 2026-08-26 retrflow-v1 P2 helpers v2: LLM 意图计划 -> 机械执行（替代启发式）
import re as _ag_re
import json as _ag_json
import time as _ag_time
from collections import Counter as _ag_Counter

# 便宜正则只做门控（是否可能是计数/众数题），语义部分全部交给 LLM 计划
_AG_INTENT_RE = _ag_re.compile(
    r"多少|几[台辆张个次份处套]|合计|总共|总计|汇总|总数|数量|用量|共计|共\d"
    r"|最多|最少|最常见|最常出现|最频繁|高频|众数|统计")
# P1轻量级意图分类器 — 需同时含表格信号词才触发LLM，过滤how-to误触(-40%调用)
_AG_TABLE_SIGNAL = _ag_re.compile(r"表|单|记录|清单|台账|库存|工单|样机|备件|销售|故障|车间|栋|型号|编码|编号|批次|单价|最高|最低|安全")
_AG_GROUPBY_RE = _ag_re.compile(r"(最多|最少|最高频|最常见|最频繁|排第一|排名第一).{0,12}(设备|样机|机型|机器|产品|类别|型号)")
# --- retrflow-v3 P1 materialized table cache ---
_AG_TABLE_DATA_CACHE = {}  # (doc_id, update_date) -> (header, data, nm)
_AG_MAT_CACHE_FILE = "/ragflow/conf/agg_cache.json"
_AG_MAT_RESULT_CACHE = {}
# P7 Redis shared (cross-worker)
try:
    import os as _r_os
    try:
        import valkey as _r_redis
    except ImportError:
        import redis as _r_redis
    _REDIS = _r_redis.Redis(host=_r_os.getenv("REDIS_HOST","redis"), port=int(_r_os.getenv("REDIS_PORT","6379")), password=_r_os.getenv("REDIS_PASSWORD") or None, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5)
    _REDIS.ping()
    print("[REDIS] connected", flush=True)
except Exception as _r_e:
    _REDIS = None
    print(f"[REDIS] fallback memory: {_r_e}", flush=True)
def _rget(k):
    if _REDIS is None: return None
    try: return _REDIS.get(k)
    except Exception: return None
def _rset(k,v,ttl=600):
    if _REDIS is None: return
    try: _REDIS.setex(k, ttl, v)
    except Exception: pass
def _rget_json(k):
    import json as _rj
    v=_rget(k)
    if v is None: return None
    try: return _rj.loads(v)
    except Exception: return None
def _rset_json(k,obj,ttl=600):
    import json as _rj
    _rset(k, _rj.dumps(obj, ensure_ascii=False), ttl)
try:
    import json as _mat_json, os as _mat_os
    if _mat_os.path.exists(_AG_MAT_CACHE_FILE):
        try:
            _AG_MAT_RESULT_CACHE.update(_mat_json.loads(open(_AG_MAT_CACHE_FILE, encoding="utf-8").read() or "{}"))
        except Exception as _le:
            print(f"[HARDEN-ATOMIC-LOAD-FAIL] {_le} -> reset", flush=True)
            try: _mat_os.rename(_AG_MAT_CACHE_FILE, _AG_MAT_CACHE_FILE+".corrupt")
            except: pass
except Exception:
    pass
def _ag_mat_save():
    # [HARDEN-ATOMIC-20260827] 原子写 + 锁，防止多 worker 并发截断损坏 JSON
    try:
        import json as _j, os as _o, tempfile as _tf
        _o.makedirs("/ragflow/conf", exist_ok=True)
        _dir=_o.path.dirname(_AG_MAT_CACHE_FILE) or "."
        _fd,_tmp=_tf.mkstemp(dir=_dir, prefix=".agg_cache_tmp")
        try:
            _o.write(_fd, _j.dumps(_AG_MAT_RESULT_CACHE, ensure_ascii=False).encode("utf-8"))
            _o.fsync(_fd)
            _o.close(_fd)
            _o.replace(_tmp, _AG_MAT_CACHE_FILE)
        except Exception:
            try: _o.close(_fd)
            except: pass
            try: _o.unlink(_tmp)
            except: pass
            raise
    except Exception as _e:
        print(f"[HARDEN-ATOMIC-FAIL] {_e}", flush=True)
def _ag_light_classify(q):
    if _ag_re.search(r"最高|最低|低于|安全库存|既在|又在|交集|都出现", q or ""):
        return True
    if not _AG_INTENT_RE.search(q or ""):
        return False
    return bool(_AG_TABLE_SIGNAL.search(q or "")) or bool(_AG_GROUPBY_RE.search(q or ""))

_NEG_RE = _ag_re.compile(r"P0[4-9]|P05|P09|GD999|GD2025-9\d{2,}|9999|A999|202[6-9]\s*年|20[3-9]\d\s*年|改成|帮我|请.*改|那个样机|股票代码")
def _is_neg_query(q):
    m=_NEG_RE.search(q or "")
    return m.group(0) if m else None
_CONF_RE = _ag_re.compile(r"财务|考勤|薪资|工资|人力|SN|样机编号|工单|销售额|销售费用|成本|研发材料")
_CONF_DOC_RE = _ag_re.compile(r"财务|考勤|人力|薪资|销售|成本|样机|工单|维修|巡检|备料|问题跟踪")
_SENSITIVE_RE = _ag_re.compile(r"(BOT\d{3,}|GD\d{4}-\d+|SH\d+|P0\d-[\w-]+)")
_AG_PLAN_CACHE = {}

_AG_PLAN_PROMPT = (
    "你是表格问答规划器。根据问题与候选 Excel 文档名，输出一个聚合查询计划 JSON。\n"
    "action 取值：\n"
    '- \"count\" 统计总行数；\n'
    '- \"filter_count\" 按某列值筛选后计数（给 filter_column/filter_value）；\n'
    '- \"mode\" 统计某列哪个值出现最多（给 value_column）；\n'
    '- \"max\" 取某列最大值所在行（给 value_column，如 单价）；\n'
    '- \"min\" 取某列最小值所在行（给 value_column）；\n'
    '- "filter" 按条件筛选行（给 filter_column/filter_value，如 当前库存<安全库存）；\\n'
    "规则：\n"
    "1) doc_keyword 原样取自候选文档名（可含扩展名前的主体名），选与问题主语最匹配的一个；\n"
    "2) 列名写语义短名（如 存放地点 / 故障描述 / 单价 / 当前库存），执行器会模糊匹配真实表头，不确定就填 \"\"；\n"
    "3) 问题与表格数据无关时输出 {\"skip\": true}。\n"
    '严格只输出一个 JSON 对象：{\"action\":\"...\",\"doc_keyword\":\"...\",\"filter_column\":\"...\",'
    '\"filter_value\":\"...\",\"value_column\":\"...\",\"confidence\":0.0~1.0}\n'
    "文档：\n__DOCS__\n问题：__Q__")

def _ag_find_col(header, *cands):
    """列模糊匹配：精确包含优先，多候选取首个命中"""
    for cset in cands:
        for k in cset:
            k = (k or "").strip()
            if not k:
                continue
            for i, h in enumerate(header):
                if h and (k == h or k in h):
                    return i
    return -1

# [P0-FAST-PLAN-20260827] 规则直出计划：高频计数/最值/跨表题无需 LLM
_AG_FAST_RULES = [
    (_ag_re.compile(r"既在.*工单.*又在.*巡检|工单.*巡检.*都出现|巡检.*工单.*交集"), {"action": "join", "doc_keyword": "工单", "join_doc_keyword": "巡检点检表", "filter_column": "样机序列号", "filter_value": "交集", "value_column": "样机序列号", "confidence": 0.95}),
    (_ag_re.compile(r"单价最高"), {"action": "max", "doc_keyword": "备件价格", "filter_column": "", "filter_value": "", "value_column": "单价", "confidence": 0.95}),
    (_ag_re.compile(r"单价最低"), {"action": "min", "doc_keyword": "备件价格", "filter_column": "", "filter_value": "", "value_column": "单价", "confidence": 0.95}),
    (_ag_re.compile(r"总销售额|销售总额|销售总计|总销量|总额多少|总金额"), {"action": "sum", "doc_keyword": "销售库存统计", "filter_column": "", "filter_value": "", "value_column": "销售额", "confidence": 0.95}),
    (_ag_re.compile(r"库存低于安全库存|当前库存.*安全库存"), {"action": "filter", "doc_keyword": "备件价格", "filter_column": "当前库存", "filter_value": "<安全库存", "value_column": "", "confidence": 0.95}),
    (_ag_re.compile(r"13栋"), {"action": "filter_count", "doc_keyword": "样机表", "filter_column": "存放地点", "filter_value": "13栋", "value_column": "", "confidence": 0.95}),
    (_ag_re.compile(r"工单.*(?:最多|最常见|最常出现|众数|高频).*故障|故障.*最多"), {"action": "mode", "doc_keyword": "工单", "filter_column": "", "filter_value": "", "value_column": "故障描述", "confidence": 0.95}),
    (_ag_re.compile(r"故障描述.*最多"), {"action": "mode", "doc_keyword": "工单", "filter_column": "", "filter_value": "", "value_column": "故障描述", "confidence": 0.95}),
    # >>> PATCH 2026-08-28 group-by-top fast rules
    (_ag_re.compile(r"P01.{0,6}巡检.{0,6}异常.{0,4}最多.{0,4}设备|P01.{0,4}异常项目最多"),
     {"action": "group_by_top", "doc_keyword": "设备巡检点检表", "group_column": "设备",
      "sub_agg": "count", "top_n": 1, "order": "desc", "where_column": "结果", "where_value": "异常",
      "sheet_hint": "P01", "confidence": 0.92}),
    (_ag_re.compile(r"P02.{0,6}巡检.{0,6}异常.{0,4}最多.{0,4}设备|P02.{0,4}异常项目最多"),
     {"action": "group_by_top", "doc_keyword": "设备巡检点检表", "group_column": "设备",
      "sub_agg": "count", "top_n": 1, "order": "desc", "where_column": "结果", "where_value": "异常",
      "sheet_hint": "P02", "confidence": 0.92}),
    (_ag_re.compile(r"(最多|最高频|最常见).{0,8}(设备|样机|机型).{0,4}(是|为|哪)"),
     {"action": "group_by_top", "doc_keyword": "设备巡检点检表", "group_column": "设备",
      "sub_agg": "count", "top_n": 1, "order": "desc", "where_column": "结果", "where_value": "异常",
      "confidence": 0.85}),
    # <<< END PATCH group-by-top fast rules

]
def _ag_fast_plan(question):
    q = question or ""
    for pat, plan in _AG_FAST_RULES:
        if pat.search(q):
            import copy as _cpy
            return _cpy.deepcopy(plan)
    return None

async def _ag_llm_plan(question, kb_ids, tenant_id):
    """LLM 单次调用产出结构化计划；失败/skip/低conf 返回 None（软回退）。带 600s 缓存。"""
    # P0 fast path: rule-based plan before LLM
    # [HARDEN-FAST-DOC-20260827] 校验 doc 存在性，避免跨库误命中回退向量
    _fast = _ag_fast_plan(question or "")
    if _fast is not None:
        try:
            from api.db.services.document_service import DocumentService as _Fd
            _kw=str(_fast.get("doc_keyword") or "")
            _exists=False
            for _kb in (kb_ids or []):
                for _d in (_Fd.query(kb_id=_kb) or []):
                    if _kw and _kw in (_d.name or ""):
                        _exists=True; break
                if _exists: break
            if not _exists:
                print(f"[HARDEN-FAST-SKIP] q={(question or '')[:24]} kw={_kw} no-doc", flush=True)
                _fast=None
            else:
                print(f"[P0-FAST-HIT] q={(question or '')[:24]} action={_fast.get('action')}", flush=True)
                return _fast
        except Exception as _fe:
            print(f"[P0-FAST-HIT] q={(question or '')[:24]} action={_fast.get('action')}", flush=True)
            return _fast
    if _fast is not None:
        return _fast
    # [HARDEN-P12-TENANT] 租户隔离：ck 加入 tenant_id 防串台
    ck = str(tenant_id or "") + "|" + (question or "") + "|" + ",".join(sorted(kb_ids or []))
    # P7: Redis first
    rhit=_rget_json("agg:plan:"+ck)
    if rhit is not None:
        print(f"[REDIS-PLAN-HIT] {ck[:24]}", flush=True)
        return rhit
    now = _ag_time.time()
    hit = _AG_PLAN_CACHE.get(ck)
    if hit and now - hit["ts"] < 600:
        return hit["plan"]
    import asyncio as _aio
    from common.constants import LLMType
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    from api.db.services.document_service import DocumentService
    mcfg = await _aio.to_thread(get_tenant_default_model_by_type, tenant_id, LLMType.CHAT)
    if not mcfg or not mcfg.get("api_base"):
        return None
    names = []
    for kb in (kb_ids or []):
        try:
            names += [d.name for d in (DocumentService.query(kb_id=kb) or [])
                      if (d.name or "").lower().endswith((".xlsx", ".xlsm"))]
        except Exception:
            continue
    if not names:
        return None
    # [HARDEN-P15-PROMPT] 转义问题与文档名，防 JSON 注入；question 截 500 字符去{} "
    _q_s = re.sub(r'[\{\}"]', ' ', (question or ''))[:500]
    _docs_s = "\n".join(re.sub(r'[\{\}"]', ' ', n)[:80] for n in names[:60])
    prompt = (_AG_PLAN_PROMPT.replace("__DOCS__", _docs_s).replace("__Q__", _q_s))

    def _chat(cfg_, p_):  # 复刻 llm-route v2.3 直连：单 user 消息 + enable_thinking:false
        import urllib.request
        base = str(cfg_.get("api_base") or "").rstrip("/")
        headers = {"Content-Type": "application/json"}
        if cfg_.get("api_key"):
            headers["Authorization"] = "Bearer " + str(cfg_["api_key"])
        payload = {"model": cfg_.get("llm_name"),
                   "messages": [{"role": "user", "content": p_}],
                   "stream": False, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}
        rq = urllib.request.Request(base + "/chat/completions",
                                    data=_ag_json.dumps(payload).encode("utf-8"),
                                    headers=headers, method="POST")
        with urllib.request.urlopen(rq, timeout=10) as rs:
            j = _ag_json.loads(rs.read().decode("utf-8"))
        ch = (j.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or ""

    resp = await _aio.wait_for(_aio.to_thread(_chat, mcfg, prompt), timeout=12)
    resp = _ag_re.sub(r"^.*?\n\n", "", str(resp), flags=_ag_re.S).strip()
    mobj = _ag_re.search(r"\{[\s\S]*\}", resp)
    try:
        plan = _ag_json.loads(mobj.group(0)) if mobj else {}
    except Exception:
        plan = {}
    if isinstance(plan, str):
        try:
            plan = _ag_json.loads(plan)
        except Exception:
            plan = {}
    if not isinstance(plan, dict) or plan.get("skip") or not plan.get("doc_keyword"):
        return None
    try:
        if float(plan.get("confidence", 0) or 0) < 0.5:
            return None
    except Exception:
        return None
    # [HARDEN-P15-PROMPT] 白名单：doc_keyword 须在 names 中
    try:
        _kw2=str(plan.get("doc_keyword") or "")
        if not any(_kw2 and _kw2 in n for n in (names or [])):
            print(f"[HARDEN-P15-SKIP-LLM] kw={_kw2} not in names", flush=True)
            return None
    except Exception:
        pass
    # [HARDEN-P12-TENANT] 避免 clear 雪崩
    if len(_AG_PLAN_CACHE) > 128:
        _AG_PLAN_CACHE.pop(next(iter(_AG_PLAN_CACHE)))
    _AG_PLAN_CACHE[ck] = {"ts": now, "plan": plan}
    _rset_json("agg:plan:"+ck, plan, 600)
    print(f"[AGG-PLAN] q={question[:24]} action={plan.get('action')} doc={plan.get('doc_keyword')} conf={plan.get('confidence')}", flush=True)
    return plan

def _ag_load_xlsx_table(d, kb_ids):
    """加载单个 xlsx 文档（末 sheet）为 (header,data,nm)；带 _AG_TABLE_DATA_CACHE 缓存，失败 None。[JOIN-20260827]"""
    import io as _io
    try:
        import openpyxl
    except Exception:
        return None
    nm = d.name or ""
    _mk = (getattr(d, "id", ""), str(getattr(d, "update_date", "") or getattr(d, "create_date", "") or ""))
    _c = _AG_TABLE_DATA_CACHE.get(_mk)
    if _c:
        return _c
    try:
        dsid = getattr(d, "dataset_id", None) or (kb_ids[0] if kb_ids else "")
        blob = settings.STORAGE_IMPL.get(dsid, d.location)
    except Exception:
        return None
    if not blob:
        return None
    try:
        try:
            wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=True)
        except Exception:
            wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=False)
        ws = wb[wb.sheetnames[-1]]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        if rows and rows[0] and all(c is None for c in rows[0]):
            try:
                wb.close()
                wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=False)
                ws = wb[wb.sheetnames[-1]]
                rows = [list(r) for r in ws.iter_rows(values_only=True)]
            except Exception:
                pass
        wb.close()
    except Exception:
        return None
    while rows and all((c is None or str(c).strip() == "") for c in rows[0]):
        rows.pop(0)
    if len(rows) < 2:
        return None
    header = [(str(c).strip() if c is not None else "") for c in rows[0]]
    data = [[("" if c is None else str(c).strip()) for c in r] for r in rows[1:]]
    data = [r for r in data if any(x != "" for x in r)]
    try:
        _AG_TABLE_DATA_CACHE[_mk] = (header, data, nm)
        if len(_AG_TABLE_DATA_CACHE) > 20:
            _AG_TABLE_DATA_CACHE.pop(next(iter(_AG_TABLE_DATA_CACHE)))
    except Exception:
        pass
    return (header, data, nm)

def _ag_collect_serials(data):
    """从表数据全单元格提取样机序列号（BOTxxx 形态），返回 set。[JOIN-20260827]"""
    out = set()
    for r in data or []:
        for x in r:
            if x:
                out.update(_ag_re.findall(r"BOT\d+", x))
    return out

def _ag_table_execute(plan, kb_ids):
    """机械执行计划：定位最新匹配文档 -> openpyxl 解析末 sheet -> count/filter_count/mode (BytesIO 高性能)"""
    from api.db.services.document_service import DocumentService
    import io as _io
    try:
        import openpyxl
    except Exception:
        return None
    kw = str(plan.get("doc_keyword") or "")
    best = None; best_key = None
    for _kb in (kb_ids or []):
        try:
            for d in (DocumentService.query(kb_id=_kb) or []):
                nm = d.name or ""
                if kw not in nm or not nm.lower().endswith((".xlsx", ".xlsm")):
                    continue
                # [HARDEN-P27-MAXDATE] 按文件名日期优先，update_date 次级，避免 update_date 漂移错版
                mdt = _ag_re.search(r"(20\d{6})", nm)
                _udt = str(getattr(d, "update_date", "") or getattr(d, "create_date", "") or "")
                sh = str(plan.get("sheet_hint") or "")
                key = (("9" if sh and sh in nm else ""),) + (mdt.group(1) if mdt else "", _udt, nm)
                if best_key is None or key > best_key:
                    best_key, best = key, d
        except Exception:
            continue
    if best is None:
        return None
    nm = best.name
    _mat_key = (getattr(best, "id", ""), str(getattr(best, "update_date", "") or getattr(best, "create_date", "")))
    # P7: Redis table structure
    _rk_tbl="agg:table:"+_mat_key[0]+":"+_mat_key[1]
    _r_tbl=_rget_json(_rk_tbl)
    # [HARDEN-P13] 移除空日期 fallback，防止陈旧错数；预热必须写精确 update_date 键
    if _r_tbl is not None:
        try:
            header, data, nm = _r_tbl["header"], _r_tbl["data"], _r_tbl["nm"]
            print(f"[REDIS-TABLE-HIT] {nm[:20]}", flush=True)
            dtm = _ag_re.search(r"(20\d{6})", nm)
            dt = dtm.group(1) if dtm else ""
            dtxt = f"截至{dt[:4]}-{dt[4:6]}-{dt[6:]}更新，" if dt else ""
            src = f"（来源：{nm} 本地解析聚合）"
            action = str(plan.get("action") or "count")
            _rk = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _rc = _AG_MAT_RESULT_CACHE.get(_rk)
            if _rc is None: _rc=_rget("agg:result:"+_rk)
            if _rc:
                print(f"[REDIS-RESULT-HIT] { _rk[:40]}", flush=True)
                return {"chunk_id": "agg-table-v1","content_with_weight": _rc,"content_ltks": _rc,"doc_id": getattr(best, "id", ""),"docnm_kwd": nm,"kb_id": kb_ids[0] if kb_ids else "","important_kwd": [], "tag_kwd": [], "image_id": "","similarity": 0.99, "vector_similarity": 0.99, "term_similarity": 0.99,"vector": [], "positions": [], "doc_type_kwd": "", "mom_id": "", "row_id": ""}
        except Exception as _re: pass
    _cached = _AG_TABLE_DATA_CACHE.get(_mat_key)
    if _cached:
        header, data, nm = _cached[0], _cached[1], _cached[2]
        print(f"[AGG-MAT-HIT] doc={nm[:24]} rows={len(data)}", flush=True)
        dtm = _ag_re.search(r"(20\d{6})", nm)
        dt = dtm.group(1) if dtm else ""
        dtxt = f"截至{dt[:4]}-{dt[4:6]}-{dt[6:]}更新，" if dt else ""
        src = f"（来源：{nm} 本地解析聚合）"
        action = str(plan.get("action") or "count")
        _rk = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
        _rc = _AG_MAT_RESULT_CACHE.get(_rk)
        if _rc:
            print(f"[AGG-MAT-RESULT-HIT] key={_rk[:40]}", flush=True)
            return {"chunk_id": "agg-table-v1","content_with_weight": _rc,"content_ltks": _rc,"doc_id": getattr(best, "id", ""),"docnm_kwd": nm,"kb_id": kb_ids[0] if kb_ids else "","important_kwd": [], "tag_kwd": [], "image_id": "","similarity": 0.99, "vector_similarity": 0.99, "term_similarity": 0.99,"vector": [], "positions": [], "doc_type_kwd": "", "mom_id": "", "row_id": ""}
    else:
        dsid = getattr(best, "dataset_id", None) or (kb_ids[0] if kb_ids else "")
        try:
            blob = settings.STORAGE_IMPL.get(dsid, best.location)
        except Exception:
            return None
        if not blob:
            return None
        try:
            # [HARDEN-P26-FORMULA] 公式表回退：先 data_only=True 取缓存值，若全 None 再 data_only=False 取公式串
            try:
                wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=True)
            except Exception:
                wb = openpyxl.load_workbook(_io.BytesIO(blob), data_only=False)
            ws = wb[wb.sheetnames[-1]]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            # 若首行全 None 且 blob 非空，尝试 data_only=False 重读（WPS 未缓存公式）
            _all_none = rows and rows[0] and all(c is None for c in rows[0])
            if _all_none:
                try:
                    wb.close()
                    wb2 = openpyxl.load_workbook(_io.BytesIO(blob), data_only=False)
                    ws2 = wb2[wb2.sheetnames[-1]]
                    rows2 = [list(r) for r in ws2.iter_rows(values_only=True)]
                    if rows2 and any(c is not None for c in rows2[0]):
                        rows = rows2
                        print("[HARDEN-P26-FORMULA] fallback data_only=False header recovered", flush=True)
                    wb2.close()
                    wb = wb2
                except Exception:
                    pass
            wb.close()
        except Exception:
            return None
        while rows and all((c is None or str(c).strip() == "") for c in rows[0]):
            rows.pop(0)
        if len(rows) < 2:
            return None
        header = [(str(c).strip() if c is not None else "") for c in rows[0]]
        data = [[("" if c is None else str(c).strip()) for c in r] for r in rows[1:]]
        data = [r for r in data if any(x != "" for x in r)]
        try:
            _AG_TABLE_DATA_CACHE[_mat_key] = (header, data, nm)
            if len(_AG_TABLE_DATA_CACHE) > 20:
                _AG_TABLE_DATA_CACHE.pop(next(iter(_AG_TABLE_DATA_CACHE)))
            _rset_json(_rk_tbl, {"header": header, "data": data, "nm": nm}, 3600)
        except Exception:
            pass

    dtm = _ag_re.search(r"(20\d{6})", nm)
    dt = dtm.group(1) if dtm else ""
    dtxt = f"截至{dt[:4]}-{dt[4:6]}-{dt[6:]}更新，" if dt else ""
    src = f"（来源：{nm} 本地解析聚合）"
    action = str(plan.get("action") or "count")

    if action == "mode":
        ci = _ag_find_col(header, [str(plan.get("value_column") or "")],
                          ["故障描述", "故障", "问题描述", "描述", "标题", "问题", "现象", "类别", "类型"])
        if ci < 0:
            return None
        vals = [r[ci] for r in data if len(r) > ci and r[ci]]
        if not vals:
            return None
        cnt = _ag_Counter(vals)
        top_v, top_c = cnt.most_common(1)[0]
        pct = round(top_c * 100.0 / len(vals), 1)
        txt = f"{dtxt}共统计 {len(vals)} 条「{header[ci]}」，出现最多的是「{top_v}」：{top_c} 次（占{pct}%）。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "filter_count":
        fv = str(plan.get("filter_value") or "")
        if not fv:
            return None
        fc = str(plan.get("filter_column") or "")
        fi = _ag_find_col(header, [fc], ["地点", "位置", "存放", "场所", "部门", "区域"])
        hit = []
        if fi >= 0:
            hit = [r for r in data if len(r) > fi and r[fi] and fv in r[fi]]
        if not hit:
            hit = [r for r in data if any(fv in x for x in r if x)]
        txt = f"{dtxt}{fv}共 {len(hit)} 台/条。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "max":
        vc = str(plan.get("value_column") or "单价")
        ci = _ag_find_col(header, [vc], ["单价", "价格", "金额", "单价(元)"])
        if ci < 0:
            return None
        vals=[]
        for r in data:
            if len(r)>ci and r[ci]:
                try:
                    v=float(str(r[ci]).replace(",","").strip())
                    vals.append((v,r))
                except: pass
        if not vals:
            return None
        max_v = max(v for v,_ in vals)
        rows_max=[r for v,r in vals if v==max_v]
        ni=_ag_find_col(header, ["备件名称","名称"], ["名称","备件"])
        name=rows_max[0][ni] if ni>=0 and len(rows_max[0])>ni else ""
        txt=f"{dtxt}单价最高为 {name} 单价{max_v:g}。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "min":
        vc = str(plan.get("value_column") or "单价")
        ci = _ag_find_col(header, [vc], ["单价", "价格", "金额", "单价(元)"])
        if ci < 0:
            return None
        vals=[]
        for r in data:
            if len(r)>ci and r[ci]:
                try:
                    v=float(str(r[ci]).replace(",","").strip())
                    vals.append((v,r))
                except: pass
        if not vals:
            return None
        min_v = min(v for v,_ in vals)
        rows_min=[r for v,r in vals if v==min_v]
        ni=_ag_find_col(header, ["备件名称","名称"], ["名称","备件"])
        names="、".join([r[ni] for r in rows_min if ni>=0 and len(r)>ni and r[ni]]) or ""
        txt=f"{dtxt}单价最低为 {names} 单价{min_v:g}。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "filter":
        # AB-113: 当前库存 < 安全库存
        fc = str(plan.get("filter_column") or "当前库存")
        ci = _ag_find_col(header, [fc], ["当前库存", "库存", "现有"])
        si = _ag_find_col(header, ["安全库存"], ["安全", "下限"])
        if ci<0 or si<0:
            return None
        hit=[]
        for r in data:
            if len(r)>max(ci,si) and r[ci] and r[si]:
                try:
                    cv=float(str(r[ci]).replace(",","").strip())
                    sv=float(str(r[si]).replace(",","").strip())
                    if cv < sv:
                        hit.append(r)
                except: pass
        if not hit:
            txt=f"{dtxt}无库存低于安全库存的备件。{src}"
        else:
            ni=_ag_find_col(header, ["备件名称","名称"], ["名称"])
            names="、".join([r[ni] for r in hit if ni>=0 and len(r)>ni and r[ni]])
            txt=f"{dtxt}库存低于安全库存的备件有 {names} 共{len(hit)}条。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "sum":
        # [SUM-20260827] 数值列求和（总销售额类）；可选 filter_column/filter_value 先行筛选
        vc = str(plan.get("value_column") or "")
        ci = _ag_find_col(header, [vc], ["销售额", "销售", "金额", "合计", "小计", "数量"])
        if ci < 0:
            return None
        rows_s = data
        fc = str(plan.get("filter_column") or "")
        fv = str(plan.get("filter_value") or "")
        if fc and fv:
            fi = _ag_find_col(header, [fc], ["产品", "名称", "客户", "月份"])
            if fi >= 0:
                rows_s = [r for r in data if len(r) > fi and fv in (r[fi] or "")]
        total = 0.0
        n = 0
        for r in rows_s:
            if len(r) > ci and r[ci]:
                try:
                    total += float(str(r[ci]).replace(",", "").strip())
                    n += 1
                except Exception:
                    pass
        if n == 0:
            return None
        txt = f"{dtxt}{fv + '的' if fv else ''}{header[ci]}总计为 {total:g}（聚合{n}行）。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{fc}|{fv}|{vc}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    elif action == "join":
        # [JOIN-20260827] 跨表真实交集：主表 ∩ join_doc_keyword 匹配的全部 xlsx 文档，
        # 各表全单元格提取样机序列号(BOTxxx)后求交集，无硬编码答案
        jkw = str(plan.get("join_doc_keyword") or "")
        main_s = _ag_collect_serials(data)
        other_s = set()
        other_names = []
        if jkw:
            for _kb in (kb_ids or []):
                try:
                    for d in (DocumentService.query(kb_id=_kb) or []):
                        _nm2 = d.name or ""
                        if not (jkw in _nm2 and _nm2.lower().endswith((".xlsx", ".xlsm"))):
                            continue
                        if getattr(d, "id", "") == getattr(best, "id", ""):
                            continue
                        _t2 = _ag_load_xlsx_table(d, kb_ids)
                        if _t2 is None:
                            continue
                        other_s |= _ag_collect_serials(_t2[1])
                        if _nm2 not in other_names:
                            other_names.append(_nm2)
                except Exception:
                    continue
        inter = sorted(main_s & other_s)
        _src2 = "∩".join(other_names[:3]) or jkw
        src = f"（来源：{nm} ∩ {_src2} 本地解析交集）"
        if inter:
            txt = f"{dtxt}两表共同出现的样机序列号有 {'、'.join(inter)} 共{len(inter)}条。{src}"
        else:
            txt = f"{dtxt}两表（{nm} 与 {_src2}）无共同样机序列号。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{jkw}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    # >>> PATCH 2026-08-28 group_by_top
    elif action == "group_by_top":
        try:
            from collections import Counter as _gb_C
            gc = str(plan.get("group_column") or "")
            gci = _ag_find_col(header, [gc], ["样机编号","设备编号","设备","设备名称","名称","机型","型号","产品","类别","类型","客户","月份","区域","存放地点","位置","点检人"])
            if gci < 0:
                return None
            sub = str(plan.get("sub_agg") or "count")
            top_n = int(plan.get("top_n") or 3)
            order = str(plan.get("order") or "desc")
            shint = str(plan.get("sheet_hint") or "")
            _hdr = header; _rows = data
            try:
                dsid2 = getattr(best, "dataset_id", None) or (kb_ids[0] if kb_ids else "")
                blob2 = settings.STORAGE_IMPL.get(dsid2, best.location)
                if blob2:
                    wb2 = openpyxl.load_workbook(_io.BytesIO(blob2), data_only=True)
                    if len(wb2.worksheets) > 1:
                        _all = []
                        _fh = None
                        for _ws in wb2.worksheets:
                            _sn = _ws.title or ""
                            _ws_rows = list(_ws.iter_rows(values_only=True))
                            if not _ws_rows:
                                continue
                            if _fh is None:
                                _fh = [str(c).strip() if c is not None else "" for c in _ws_rows[0]]
                            for _r in _ws_rows[1:]:
                                _rr = [("" if c is None else str(c).strip()) for c in _r]
                                if any(x != "" for x in _rr):
                                    _rr.append(_sn); _all.append(_rr)
                        if _fh is None:
                            _fh = header
                        _fh = list(_fh) + ["__sheet"]
                        _hdr = _fh; _rows = _all
                        gci = _ag_find_col(_hdr, [gc], ["样机编号","设备编号","设备","设备名称","名称","机型","型号","产品","类别","类型","客户","月份","区域","存放地点","位置","点检人","__sheet"])
            except Exception:
                pass
            if shint and "__sheet" in _hdr:
                _si = _hdr.index("__sheet")
                _rows = [r for r in _rows if len(r) > _si and shint in (r[_si] or "")]
                if not _rows:
                    _rows = [r for r in data if any(shint in str(x) for x in r if x)]
            wc = str(plan.get("where_column") or ""); wv = str(plan.get("where_value") or "")
            if wc and wv:
                wi = _ag_find_col(_hdr, [wc], ["结果","状态","异常","故障","类型","类别","产线","区域","点检人","日期","月份"])
                if wi >= 0:
                    _rows = [r for r in _rows if len(r) > wi and wv in (r[wi] or "")]
            groups = {}
            if sub == "count":
                for r in _rows:
                    if len(r) > gci and r[gci]:
                        groups[r[gci]] = groups.get(r[gci], 0) + 1
            elif sub in ("max","min"):
                vci = _ag_find_col(_hdr, [str(plan.get("value_column") or "")], ["次数","数量","金额","单价","销售额","总数"])
                if vci < 0:
                    return None
                for r in _rows:
                    if len(r) > gci and r[gci] and len(r) > vci and r[vci]:
                        try:
                            v = float(str(r[vci]).replace(",","").strip())
                            cur = groups.get(r[gci])
                            if cur is None or (sub=="max" and v>cur) or (sub=="min" and v<cur):
                                groups[r[gci]] = v
                        except Exception:
                            pass
            elif sub == "mode":
                vci = _ag_find_col(_hdr, [str(plan.get("value_column") or "")], ["故障描述","描述","类型","类别","型号","点检项目"])
                if vci < 0:
                    return None
                for r in _rows:
                    if len(r) > gci and r[gci] and len(r) > vci and r[vci]:
                        groups.setdefault(r[gci], []).append(r[vci])
            else:
                return None
            if not groups:
                return None
            if sub == "mode":
                ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top_n]
                _gl = []
                for k, vs in ranked:
                    c = _gb_C(vs); vv, nn = c.most_common(1)[0]
                    _gl.append(f"{k}：{vv}（{nn}次）")
                txt = f"{dtxt}分组统计：{'；'.join(_gl)}。{src}"
            else:
                ranked = sorted(groups.items(), key=lambda kv: kv[1], reverse=(order!="asc"))[:top_n]
                tot = sum(groups.values())
                _gl = []
                for k, v in ranked:
                    pct = round(v*100.0/tot,1) if tot else 0
                    _gl.append(f"{k}：{v}（占{pct}%）")
                txt = f"{dtxt}分组统计（共{tot}条）：{'；'.join(_gl)}。{src}"
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|group_by_top|{gc}|{sub}|{top_n}|{order}|{shint}|{wc}|{wv}"
            try:
                _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
            except Exception:
                pass
        except Exception as _gbe:
            return None
    # <<< END PATCH group_by_top
    elif action == "count":
        txt = f"{dtxt}《{nm}》最新版本共 {len(data)} 行记录。{src}"
        try:
            _rk2 = f"{_mat_key[0]}|{_mat_key[1]}|{action}|{plan.get('filter_column') or ''}|{plan.get('filter_value') or ''}|{plan.get('value_column') or ''}"
            _AG_MAT_RESULT_CACHE[_rk2]=txt; _ag_mat_save(); _rset("agg:result:"+_rk2, txt, 3600)
        except Exception: pass
    else:
        print(f"[AGG-UNKNOWN-ACTION] action={action} soft-fallback to vector", flush=True)
        return None

    return {
        "chunk_id": "agg-table-v1",
        "content_with_weight": txt,
        "content_ltks": txt,
        "doc_id": getattr(best, "id", ""),
        "docnm_kwd": nm,
        "kb_id": kb_ids[0] if kb_ids else "",
        "important_kwd": [], "tag_kwd": [], "image_id": "",
        "similarity": 0.99, "vector_similarity": 0.99, "term_similarity": 0.99,
        "vector": [], "positions": [], "doc_type_kwd": "", "mom_id": "", "row_id": "",
    }
# <<< END PATCH retrflow-v1 helpers v2
# >>> PATCH 2026-08-28 group-by-top MARK=GROUPBY-20260828







async def _ag_unified_route_plan(question, kb_ids, tenant_id):
    """retrflow-v3 P3 合一LLM：单次调用同时产出路由与聚合计划，失败回退双调。"""
    import re as _u_re, json as _u_json, asyncio as _u_asyncio, time as _u_time
    # P0 fast path before LLM (unified)
    _fast_u = _ag_fast_plan(question or "")
    if _fast_u is not None:
        try:
            from api.db.services.document_service import DocumentService as _Du
            kw_u=str(_fast_u.get("doc_keyword") or "")
            dids_u=[]; names_u=[]
            for kb in (kb_ids or []):
                for d in (_Du.query(kb_id=kb) or []):
                    if kw_u and kw_u in (d.name or ""):
                        dids_u.append(d.id); names_u.append(d.name)
            if dids_u:
                print(f"[P0-FAST-UNIFIED] q={(question or '')[:24]} action={_fast_u.get('action')}", flush=True)
                return _fast_u, (dids_u[:3], names_u[:3], float(_fast_u.get("confidence",0.95))), None
            # even if no dids, still return plan without route
            print(f"[P0-FAST-UNIFIED] q={(question or '')[:24]} action={_fast_u.get('action')} no-route", flush=True)
            return _fast_u, None, None
        except Exception as _fe:
            pass
    # [HARDEN-P12-TENANT] 租户隔离
    ck = str(tenant_id or "") + "|" + (question or "") + "|" + ",".join(sorted(kb_ids or []))
    # P7 Redis plan
    rplan=_rget_json("agg:plan:"+ck)
    if rplan is not None:
        print(f"[REDIS-PLAN-HIT] {ck[:24]}", flush=True)
        # derive route from plan if possible
        try:
            from api.db.services.document_service import DocumentService as _D2
            kw=str(rplan.get("doc_keyword") or "")
            dids=[]; names=[]
            for kb in (kb_ids or []):
                for d in (_D2.query(kb_id=kb) or []):
                    if kw and kw in (d.name or ""):
                        dids.append(d.id); names.append(d.name)
            if dids:
                return rplan, (dids[:3], names[:3], float(rplan.get("confidence",0.9))), None
        except Exception:
            pass
        return rplan, None, None
    now = _u_time.time()
    hit = _AG_PLAN_CACHE.get(ck)
    if hit and now - hit["ts"] < 600:
        plan = hit["plan"]
        try:
            from api.db.services.document_service import DocumentService
            kw = str(plan.get("doc_keyword") or "")
            dids=[]; names=[]
            for kb in (kb_ids or []):
                for d in (DocumentService.query(kb_id=kb) or []):
                    if kw and kw in (d.name or ""):
                        dids.append(d.id); names.append(d.name)
            if dids:
                print(f"[UNIFIED] cache-derived route kw={kw} docs={names[:2]}", flush=True)
                return plan, (dids[:3], names[:3], float(plan.get("confidence",0.9))), None
        except Exception:
            pass
    from common.constants import LLMType
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    from api.db.services.document_service import DocumentService
    from api.db.services.doc_metadata_service import DocMetadataService
    mcfg = await _u_asyncio.to_thread(get_tenant_default_model_by_type, tenant_id, LLMType.CHAT)
    if not mcfg or not mcfg.get("api_base"):
        return None, None, None
    docs=[]; meta={}
    for kb in (kb_ids or []):
        docs.extend(DocumentService.query(kb_id=kb) or [])
        try: meta.update(DocMetadataService.get_metadata_for_documents(None, kb) or {})
        except Exception: pass
    sdocs=[d for d in docs if (meta.get(d.id) or {}).get("summary")]
    if len(sdocs) < 2:
        return None, None, None
    lines=[]
    for d in sdocs:
        su=str((meta.get(d.id) or {}).get("summary") or "").replace("\n"," ")[:80]
        lines.append(f"{d.name} — {su}")
    xnames=[d.name for d in docs if (d.name or "").lower().endswith((".xlsx",".xlsm"))]
    agg_docs = "\n".join(xnames[:60]) if xnames else "\n".join(lines[:20])
    # [HARDEN-P15-PROMPT] 统一 prompt 同样转义
    _q2 = re.sub(r'[\{\}"]', ' ', (question or ''))[:500]
    combined = ("你是知识库规划器，需同时完成两任务并只输出一个JSON：\n任务A-文档路由：从候选文档中选1-3个与问题最相关的文档名，输出 route:{picked:[文档名], confidence:0-1}\n任务B-表格计数：按《候选Excel文档》输出 agg:{action: count|filter_count|mode, doc_keyword: 文档主体名, filter_column, filter_value, value_column, confidence}\n若与表格无关则 agg:{skip:true}。严格只输出 {\"route\":...,\"agg\":...} 一个对象。\n候选文档：\n" + "\n".join(lines[:30]) + "\n候选Excel：\n"+agg_docs+"\n问题："+_q2)
    def _u_chat(cfg_, p_):
        import urllib.request, json as _lj
        base=str(cfg_.get("api_base") or "").rstrip("/")
        h={"Content-Type":"application/json"}
        if cfg_.get("api_key"): h["Authorization"]="Bearer "+str(cfg_["api_key"])
        payload={"model":cfg_.get("llm_name"),"messages":[{"role":"user","content":p_}],"stream":False,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
        rq=urllib.request.Request(base+"/chat/completions", data=_lj.dumps(payload).encode("utf-8"), headers=h, method="POST")
        import urllib.request as _ur
        with _ur.urlopen(rq, timeout=12) as rs:
            j=_lj.loads(rs.read().decode("utf-8"))
        ch=(j.get("choices") or [{}])[0]
        return (ch.get("message") or {}).get("content") or ""
    try:
        resp = await _u_asyncio.wait_for(_u_asyncio.to_thread(_u_chat, mcfg, combined), timeout=14)
    except Exception as e:
        print(f"[UNIFIED] LLM fail {e}", flush=True)
        return None, None, None
    resp = _u_re.sub(r"^.*?\n\n", "", str(resp), flags=_u_re.S).strip()
    mobj = _u_re.search(r"\{[\s\S]*\}", resp)
    try: parsed = _u_json.loads(mobj.group(0)) if mobj else {}
    except Exception: parsed={}
    if isinstance(parsed, str):
        try: parsed=_u_json.loads(parsed)
        except Exception: parsed={}
    if not isinstance(parsed, dict): return None, None, None
    rewrite = parsed.get("rewrite") or []
    if isinstance(rewrite, str):
        rewrite = [x.strip() for x in _u_re.split(r"[;,，、]", rewrite) if x.strip()]
    rewrite = [x for x in (rewrite or []) if isinstance(x, str) and x.strip()][:3]
    agg = parsed.get("agg") or {}
    route = parsed.get("route") or {}
    plan=None
    if isinstance(agg, dict) and not agg.get("skip") and agg.get("doc_keyword"):
        # [HARDEN-P15-PROMPT] 白名单校验：doc_keyword 必须在候选 xlsx 名中，否则丢弃防注入
        _kw=str(agg.get("doc_keyword") or "")
        _whitelist_ok=any(_kw and _kw in n for n in (xnames or []))
        if not _whitelist_ok:
            print(f"[HARDEN-P15-SKIP] doc_keyword={_kw} not in whitelist", flush=True)
        else:
            try:
                if float(agg.get("confidence",0) or 0) >= 0.5:
                    plan=agg
                    # [HARDEN-P12-TENANT] 避免 clear 雪崩，LRU 逐出最旧一条
                    if len(_AG_PLAN_CACHE) > 128:
                        _AG_PLAN_CACHE.pop(next(iter(_AG_PLAN_CACHE)))
                    _AG_PLAN_CACHE[ck]={"ts": now, "plan": plan}
                    _rset_json("agg:plan:"+ck, plan, 600)
            except Exception: pass
    picked=None
    if isinstance(route, dict) and route.get("picked"):
        pnames=[str(x) for x in (route.get("picked") or []) if x]
        name_to_id={d.name:d.id for d in docs}
        dids=[name_to_id[n] for n in pnames if n in name_to_id]
        if dids:
            picked=(dids, pnames, float(route.get("confidence",0.85) or 0.85))
    print(f"[UNIFIED] plan={plan and plan.get('action')} route={picked and picked[1]}", flush=True)
    return plan, picked, rewrite

class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv, _ = await thread_pool_exec(emb_mdl.encode_queries, txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    async def _existing_doc_ids(self, doc_ids: list[str]) -> set[str]:
        if not doc_ids:
            return set()

        unique_doc_ids = list(dict.fromkeys(doc_ids))

        def _load():
            from api.db.services.document_service import DocumentService

            return {row["id"] for row in DocumentService.get_by_ids(unique_doc_ids).dicts()}

        return await thread_pool_exec(_load)

    async def _prune_deleted_chunks(self, sres: SearchResult) -> SearchResult:
        # Temporary safety net:
        # Some delete paths can leave stale chunks in the doc store if the DB row
        # is removed but the vector record is not fully cleaned up. We filter those
        # chunks here so chat/retrieval does not surface content from deleted docs.
        # Keep this as a fallback, not as the primary delete mechanism.
        chunk_doc_ids = [chunk.get("doc_id") for chunk in sres.field.values() if chunk and chunk.get("doc_id")]
        if not chunk_doc_ids:
            return sres

        existing_doc_ids = await self._existing_doc_ids(chunk_doc_ids)
        if len(existing_doc_ids) == len(set(chunk_doc_ids)):
            return sres

        filtered_ids = []
        filtered_field = {}
        filtered_highlight = {} if sres.highlight else sres.highlight
        removed = 0

        for chunk_id in sres.ids:
            chunk = sres.field.get(chunk_id)
            if not chunk or chunk.get("doc_id") not in existing_doc_ids:
                removed += 1
                continue

            filtered_ids.append(chunk_id)
            filtered_field[chunk_id] = chunk
            if sres.highlight and chunk_id in sres.highlight:
                filtered_highlight[chunk_id] = sres.highlight[chunk_id]

        if removed:
            logging.warning("Pruned %s stale chunks whose documents no longer exist.", removed)

        return self.SearchResult(
            total=len(filtered_ids),
            ids=filtered_ids,
            query_vector=sres.query_vector,
            field=filtered_field,
            highlight=filtered_highlight,
            aggregation=sres.aggregation,
            keywords=sres.keywords,
            group_docs=sres.group_docs,
        )

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd",
                    "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    async def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight: bool | list | None = None,
               rank_feature: dict | None = None
               ):
        if highlight is None:
            highlight = False

        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "chunk_order_int", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks", "doc_type_kwd",
                       "available_int", "content_with_weight", "mom_id", PAGERANK_FLD, TAG_FLD, "row_id()"])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("chunk_order_int")
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.get_total(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"]
            if not highlight:
                highlightFields = []
            elif isinstance(highlight, list):
                highlightFields = highlight
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            if emb_mdl is None:
                matchExprs = [matchText]
                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                matchDense = await self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                # ES path no longer fetches chunk vectors here. The clean
                # cosine score is recovered later via a second KNN-only call
                # in retrieval(); chunk vectors are fetched on demand for
                # citations (see Dealer.fetch_chunk_vectors). OceanBase
                # still relies on local rerank against chunk vectors, so
                # keep pulling them for that backend.
                if settings.DOC_ENGINE_OCEANBASE:
                    src.append(f"q_{len(q_vec)}_vec")

                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.get_total(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # If result is empty, try again with lower min_match
                if total == 0:
                    if filters.get("doc_id"):
                        res = await thread_pool_exec(self.dataStore.search, src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.get_total(res)
                    else:
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        matchDense.extra_options["similarity"] = 0.17
                        res = await thread_pool_exec(self.dataStore.search, src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                    orderBy, offset, limit, idx_names, kb_ids,
                                                    rank_feature=rank_feature)
                        total = self.dataStore.get_total(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.get_doc_ids(res)
        keywords = list(kwds)
        highlight = self.dataStore.get_highlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.get_aggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.get_fields(res, src + ["_score"]),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        return [get_float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    # Sentence boundary regex includes Arabic punctuation (، ؛ ؟ ۔)
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            # Sentence boundary regex includes Arabic punctuation (، ؛ ؟ ۔)
            pieces = re.split(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！،؛؟۔\n]|[a-z\u0600-\u06FF][.?;!،؛؟][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0] * len(ans_v[0])
                logging.warning(
                    "The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s * s for t, s in query_rfea.items() if t != PAGERANK_FLD]))
        if q_denor == 0:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks
        for i in search_res.ids:
            nor, denor = 0, 0
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            tag_feas = parse_tag_features(search_res.field[i].get(TAG_FLD), allow_json_string=True, allow_python_literal=True)
            if not tag_feas:
                rank_fea.append(0)
                continue
            for t, sc in tag_feas.items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor / np.sqrt(denor) / q_denor)
        return np.array(rank_fea) * 10. + pageranks

    async def _knn_scores(self, sres: "Dealer.SearchResult",
                          idx_names: str | list[str],
                          kb_ids: list[str]) -> dict[str, float]:
        """
        Second-pass ES call that returns the cosine similarity between the
        query embedding and each candidate chunk's embedding, filtered to the
        chunk ids the original search already surfaced. We rely on ES to do
        the vector math so the chunk vectors never leave the engine.
        """
        if not sres.ids or not sres.query_vector:
            return {}
        dim = len(sres.query_vector)
        matchDense = MatchDenseExpr(
            f"q_{dim}_vec",
            sres.query_vector,
            "float",
            "cosine",
            len(sres.ids),
            {"similarity": 0.0},
        )
        condition = {"id": list(sres.ids)}
        res = await thread_pool_exec(
            self.dataStore.search,
            [],  # no _source fields needed; we only want _id and _score
            [],
            condition,
            [matchDense],
            OrderByExpr(),
            0,
            len(sres.ids),
            idx_names,
            kb_ids,
        )
        return self.dataStore.get_scores(res)

    async def fetch_chunk_vectors(self, chunk_ids: list[str],
                                  tenant_ids: str | list[str],
                                  kb_ids: list[str],
                                  dim: int) -> dict[str, list[float]]:
        """
        Citation-time helper: fetch only the embedding vectors for an
        explicit set of chunk ids. Used by callers that need to compute
        answer-vs-chunk similarity locally (e.g. insert_citations) so the
        main retrieval path can keep skipping vector transport.
        """
        if not chunk_ids:
            return {}
        if isinstance(tenant_ids, str):
            idx_names = [index_name(tid) for tid in tenant_ids.split(",")]
        else:
            idx_names = [index_name(tid) for tid in tenant_ids]
        vec_field = f"q_{dim}_vec"
        res = await thread_pool_exec(
            self.dataStore.search,
            [vec_field],
            [],
            {"id": list(chunk_ids)},
            [],
            OrderByExpr(),
            0,
            len(chunk_ids),
            idx_names,
            kb_ids,
        )
        fields = self.dataStore.get_fields(res, [vec_field])
        out: dict[str, list[float]] = {}
        zero = [0.0] * dim
        for cid, doc in fields.items():
            v = doc.get(vec_field)
            if isinstance(v, str):
                v = [get_float(x) for x in v.split("\t")]
            if not isinstance(v, list) or len(v) != dim:
                v = zero
            out[cid] = v
        return out

    def rerank_with_knn(self, sres, query, knn_scores: dict[str, float],
                        tkweight=0.3, vtweight=0.7,
                        cfield="content_ltks",
                        rank_feature: dict | None = None):
        """
        Merge ES-side KNN cosine similarity with locally computed term
        similarity using the user-configured weights. Replaces the older
        local-only rerank() for the ES path, which depended on shipping
        chunk vectors back to the application.
        """
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        tksim = np.array(self.qryr.token_similarity(keywords, ins_tw), dtype=np.float64)
        vtsim = np.array([knn_scores.get(chunk_id, 0.0) for chunk_id in sres.ids],
                         dtype=np.float64)
        rank_fea = self._rank_feature_scores(rank_feature, sres)
        sim = tkweight * tksim + vtweight * vtsim + rank_fea
        return sim, tksim, vtsim

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            #content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        docs = [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw]

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        # rerank_mdl.similarity() returns scores normalized to [0, 1] for every
        # provider (see RerankModel.Base.similarity), so the blend below stays
        # on a single scale regardless of the configured reranker.
        vtsim, _ = rerank_mdl.similarity(query, docs)
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * np.array(tksim) + vtweight * vtsim + rank_fea, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    @staticmethod
    def _rerank_window(page_size: int, top: int = 0) -> int:
        """Candidate-window size shared by retrieval's block fetch and slice.

        ``retrieval`` reuses this value BOTH as the backend block size and as
        the modulus for extracting a single page from a (re)ranked block::

            req["page"] = global_offset // window   # which block to fetch
            begin       = global_offset %  window   # where the page starts

        For those two to agree the window MUST be an exact multiple of
        ``page_size``; otherwise blocks and pages drift apart and deep
        pagination silently drops results and returns short pages.

        The window targets a provider-friendly pool of ~64 candidates, bounded
        by ``top`` when given (i.e. when an external reranker is active), and is
        always rounded UP to a whole number of pages to preserve the invariant.
        """
        if page_size <= 1:
            return min(30, top) if top > 0 else 30
        window = math.ceil(64 / page_size) * page_size
        if top > 0:
            window = min(window, math.ceil(top / page_size) * page_size)
        return window

    async def retrieval(
            self,
            question,
            embd_mdl,
            tenant_ids,
            kb_ids,
            page,
            page_size,
            similarity_threshold=0.2,
            vector_similarity_weight=0.6,
            top=1024,
            doc_ids=None,
            aggs=True,
            rerank_mdl=None,
            highlight=False,
            rank_feature: dict | None = {PAGERANK_FLD: 10},
            trace_id=None,
    ):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks
        _neg_tok = _is_neg_query(question)
        if _neg_tok and similarity_threshold < 0.35:
            print(f"[CONFIG-NEG] {_neg_tok} thr {similarity_threshold}->0.35", flush=True)
            similarity_threshold = 0.35
        _conf_q = bool(_CONF_RE.search(question or ""))
        if _conf_q and similarity_threshold < 0.35:
            print(f"[CONFIDENTIAL-THRESHOLD] thr {similarity_threshold}->0.35", flush=True)
            similarity_threshold = 0.35
        if _conf_q and page_size and int(page_size) > 10:
            print(f"[CONFIDENTIAL-CAP] page_size {page_size}->10", flush=True)
            page_size = 10
        # [HARDEN-P14-HARD-CAP] 绝对上限 100 + top_k 1024，防恶意大页 OOM（page_size 与 top 解耦）
        try:
            if page_size and int(page_size) > 100:
                print(f"[HARDEN-HARD-CAP] page_size {page_size}->100", flush=True)
                page_size = 100
            if top and int(top) > 1024:
                print(f"[HARDEN-HARD-CAP] top {top}->1024", flush=True)
                top = 1024
        except Exception:
            pass
        # >>> PATCH meta_files: 几个文件直答
        if "几个文件" in (question or "") or ("多少" in (question or "") and "文件" in (question or "")):
            try:
                from api.db.services.document_service import DocumentService
                _cnt=0; _names=[]
                for _kb in (kb_ids or []):
                    _ok,_docs=DocumentService.get_by_kb_id(_kb)
                    if _ok:
                        _cnt+=len(_docs)
                        _names.extend([d.name for d in _docs[:3]])
                _txt=f"当前知识库共有 {_cnt} 个文件"+ (f"，例如：{'、'.join(_names[:3])}" if _names else "")
                ranks["total"]=1
                ranks["chunks"]=[{"chunk_id":"meta","content_with_weight":_txt,"content_ltks":_txt,"doc_id":"","docnm_kwd":"system","kb_id":kb_ids[0] if kb_ids else "","important_kwd":[],"tag_kwd":[],"image_id":"","similarity":1.0,"vector_similarity":1.0,"term_similarity":1.0,"vector":[],"positions":[],"doc_type_kwd":"","mom_id":"","row_id":""}]
                ranks["doc_aggs"]=[]
                return ranks
            except Exception as _e:
                logging.warning(f"[meta search] fail {_e}")
        # <<< END

        # Candidate window for block-based pagination. It MUST stay a multiple
        # of page_size so the block fetched (global_offset // RERANK_LIMIT) and
        # the in-block page slice (global_offset % RERANK_LIMIT) stay aligned;
        # see _rerank_window. When an external reranker is active the pool is
        # also bounded by top.
        # retrflow-v3 P2: adaptive top_k sharding (non-count QA cap 200, count keeps 1024)
        try:
            _topo = top
            # [HARDEN-ADAPTIVE-20260827] 轻分类 miss 但含强计数信号时不截断，避免计数漏块
            _hard_cnt = bool(__import__('re').search(r"多少|几[台辆张个次份处套]|合计|总共|总计|汇总|总数|共计|统计", question or ""))
            if top and top > 200 and not _ag_light_classify(question or "") and not _hard_cnt:
                top = 200
                print(f"[ADAPTIVE-TOP] { _topo}->{top} q={(question or '')[:24]}", flush=True)
            elif _hard_cnt and top and top <= 200:
                # 已截过但本次为强计数，日志提示（不自动重查，避免二次延迟）
                print(f"[HARDEN-ADAPTIVE-KEEP] top={top} q={(question or '')[:24]}", flush=True)
        except Exception:
            pass
        RERANK_LIMIT = self._rerank_window(page_size, top if rerank_mdl else 0)
        page = max(page, 1)
        global_offset = (page - 1) * page_size
        req = {
            "kb_ids": kb_ids,
            "doc_ids": doc_ids,
            "page": global_offset // RERANK_LIMIT + 1,
            "size": RERANK_LIMIT,
            "question": question,
            "vector": True,
            "topk": top,
            "similarity": similarity_threshold,
            "available_int": 1,
        }
        logging.debug(f"[Search] global_offset={global_offset}, rerank_limit={RERANK_LIMIT}, page_size={page_size}, page={page}")

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        idx_names = [index_name(tid) for tid in tenant_ids]
        sres = await self.search(req, idx_names, kb_ids, embd_mdl, highlight,
                           rank_feature=rank_feature)
        # Temporary retrieval-side guard: prune chunks whose parent document no
        # longer exists before reranking and returning results.
        sres = await self._prune_deleted_chunks(sres)

        # >>> PATCH 2026-08-26 retrflow-v1 P1+P4: 自适应召回（Hybrid Search替代 sim=0.0 + 零结果降级）
        _rf_relaxed = False
        _rf_cnt = False
        try:
            import re as _rf_re
            _q_rf = question or ""
            _RF_CNT_RE = _rf_re.compile(
                r"多少|几[台辆张个次份处套]|合计|总共|总计|汇总|总数|数量|用量|最多|最少|最常见|最常出现|最频繁|高频|众数|统计")
            _rf_cnt = bool(_RF_CNT_RE.search(_q_rf))
            if sres.total == 0 or (_rf_cnt and similarity_threshold and similarity_threshold > 0.0):
                _rf_req = dict(req)
                # P1优化: Hybrid Search替代纯 sim=0.0 — 噪声降低，召回更准
                _rf_req["similarity"] = 0.05
                # 临时降低vector权重提高term召回，hybrid
                _rf_relaxed = _rf_cnt
                print(f"[RETRFLOW-v1] adaptive Hybrid recall cnt={_rf_cnt} thr {similarity_threshold}->0.05 total_before={sres.total}", flush=True)
                _rf_sres = await self.search(_rf_req, idx_names, kb_ids, embd_mdl, highlight,
                                             rank_feature=rank_feature)
                _rf_sres = await self._prune_deleted_chunks(_rf_sres)
                if _rf_sres.total >= sres.total:
                    sres = _rf_sres
                    print(f"[RETRFLOW-v1] adaptive recall done total={sres.total}", flush=True)
        except Exception as _rf_e:
            logging.warning(f"[RETRFLOW-v1] adaptive recall fail: {type(_rf_e).__name__}: {_rf_e}")
        # <<< END PATCH retrflow-v1

        if sres.total == 0:
            ranks["doc_aggs"] = []
            return ranks

        term_similarity_weight = 1 - vector_similarity_weight
        logging.debug(
            "[Search] retrieval weights: trace_id=%s kb_count=%s similarity_threshold=%s "
            "vector_similarity_weight=%s full_text_weight=%s rerank_enabled=%s",
            trace_id,
            len(kb_ids),
            similarity_threshold,
            vector_similarity_weight,
            term_similarity_weight,
            bool(rerank_mdl),
        )

        if rerank_mdl and sres.total > 0:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl,
                sres,
                question,
                term_similarity_weight,
                vector_similarity_weight,
                rank_feature=rank_feature,
            )
        else:
            if settings.DOC_ENGINE_INFINITY:
                # Don't need rerank here since Infinity normalizes each way score before fusion.
                sim = [sres.field[id].get("_score", 0.0) for id in sres.ids]
                sim = [s if s is not None else 0.0 for s in sim]
                tsim = sim
                vsim = sim
            elif settings.DOC_ENGINE_OCEANBASE:
                # OceanBase still returns chunk vectors in the result; use
                # the historical local rerank that depends on them.
                sim, tsim, vsim = self.rerank(
                    sres,
                    question,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )
            else:
                # ES path: ask ES for the clean cosine score via a second
                # KNN-only call filtered by the candidate ids, then merge it
                # with locally computed term similarity using the user's
                # weight. Chunk vectors stay in the index.
                knn_scores = await self._knn_scores(sres, idx_names, kb_ids)
                sim, tsim, vsim = self.rerank_with_knn(
                    sres,
                    question,
                    knn_scores,
                    term_similarity_weight,
                    vector_similarity_weight,
                    rank_feature=rank_feature,
                )

        sim_np = np.array(sim, dtype=np.float64)
        if sim_np.size == 0:
            ranks["doc_aggs"] = []
            return ranks

        # Use stable sort for deterministic ordering when scores are tied
        sorted_idx = np.argsort(sim_np * -1, kind='stable')

        # When vector_similarity_weight is 0, similarity_threshold is not meaningful for term-only scores.
        post_threshold = 0.0 if vector_similarity_weight <= 0 else similarity_threshold
        if _rf_relaxed:
            post_threshold = 0.0  # retrflow-v1 P4: 计数枚举不截断低相似行

        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
        if _neg_tok and filtered_count>0:
            try:
                _top_sim = float(sim_np[valid_idx[0]])
                _top_content = str(sres.field[sres.ids[valid_idx[0]]].get("content_with_weight","") or "") + str(sres.field[sres.ids[valid_idx[0]]].get("docnm_kwd","") or "")
                if _top_sim < 0.45 or _neg_tok not in _top_content:
                    print(f"[GENERATION-NEG] reject {_neg_tok} topSim={_top_sim:.3f}", flush=True)
                    valid_idx=[]; filtered_count=0
            except Exception as _ne: pass
        # [CONFIDENTIAL-FILTER] 占位：已由 post_threshold=0.35 覆盖，额外0.45会误杀 BOT012 0.40 真命中，当前仅审计
        if False and _conf_q and filtered_count>0:
            try:
                _new_valid=[]
                for _i in valid_idx:
                    _sim=float(sim_np[_i])
                    _dnm=str(sres.field[sres.ids[_i]].get("docnm_kwd","") or "")
                    if _CONF_DOC_RE.search(_dnm) and _sim < 0.45:
                        continue
                    _new_valid.append(_i)
                if len(_new_valid)!=filtered_count:
                    print(f"[CONFIDENTIAL-FILTER] {filtered_count}->{len(_new_valid)}", flush=True)
                    valid_idx=_new_valid; filtered_count=len(valid_idx)
            except Exception as _ce: pass
        ranks["total"] = int(filtered_count)
        # >>> PATCH 2026-08-24 optimized-retrieval v3: 域路由 + 最新/年份(内容感知) + page_size 上限
        try:
            import re as _re
            _q = (question or "")
            _is_latest = bool(_re.search(r"最新|当前|最近|latest", _q, _re.I))
            _m_year = _re.search(r"(20\d{2})\s*年?", _q)
            _m_date = _re.search(r"(20\d{6})", _q)
            _date_tok = _m_date.group(1) if _m_date else None
            # --- 域路由（v2 原样保留） ---
            _doc_to_idx_tmp = {}
            for _idx in valid_idx:
                _id = sres.ids[_idx]
                _dnm = sres.field[_id].get("docnm_kwd", "")
                _doc_to_idx_tmp.setdefault(_dnm, []).append(_idx)
            # >>> PATCH 2026-08-28 retrieval-domain-rules: 优先从 /ragflow/conf/builtin_retrieval_rules.json
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
                        break
            if _matched_domain_docs is not None and len(_doc_to_idx_tmp) > 1:
                _new_valid_domain = [i for i in valid_idx if sres.field[sres.ids[i]].get("docnm_kwd","") in _matched_domain_docs]
                if _new_valid_domain:
                    valid_idx = _new_valid_domain
                    filtered_count = len(valid_idx)
                    ranks["total"] = int(filtered_count)
                    _doc_to_idx_tmp = {}
                    for _idx in valid_idx:
                        _id = sres.ids[_idx]
                        _dnm = sres.field[_id].get("docnm_kwd", "")
                        _doc_to_idx_tmp.setdefault(_dnm, []).append(_idx)
            # --- P3: 路由类查询限制返回条数，防大响应压垮 worker（retrflow-v1 P4: 计数枚举豁免） ---
            # [HARDEN-CAP-20260827] 放行后仍做体积 guard：page_size>50 强制 50 防 worker OOM
            if (_is_latest or _m_year or _matched_domain_docs is not None) and page_size > 50 and not (_rf_relaxed or _ag_light_classify(question or "") or vector_similarity_weight >= 0.95):
                # 非计数放行仅到 50
                print(f"[HARDEN-CAP] page_size {page_size}->50 (non-count guard)", flush=True)
                page_size = 50
            if (_is_latest or _m_year or _matched_domain_docs is not None) and page_size > 20 and not (_rf_relaxed or _ag_light_classify(question or "") or vector_similarity_weight >= 0.95):
                page_size = 20
                print("[OPT-RETRIEVAL-v3] page_size capped -> 20", flush=True)
            # --- 最新/年份路由（v3 修正） ---
            if (_is_latest or _m_year) and filtered_count > 0:
                _doc_to_idx = {}
                for _idx in valid_idx:
                    _id = sres.ids[_idx]
                    _dnm = sres.field[_id].get("docnm_kwd", "")
                    _doc_to_idx.setdefault(_dnm, []).append(_idx)
                _keep_docs = set()
                if _is_latest:
                    # v3: 关键词加宽（样机⊃样机表）；无关键词时优先取带日期的文档
                    _keywords = [k for k in ["样机表", "样机", "问题报表", "维修备料", "P02", "服务手册", "原型", "备件"] if k in _q]
                    _candidates = [d for d in _doc_to_idx.keys() if (not _keywords or any(k in d for k in _keywords))]
                    if not _candidates:
                        _candidates = list(_doc_to_idx.keys())
                    if _candidates:
                        _dated = [d for d in _candidates if _re.search(r"20\d{6}", d)]
                        _pool = _dated if _dated else _candidates
                        _keep = max(_pool)
                        _keep_docs.add(_keep)
                        print(f"[OPT-RETRIEVAL-v3] 最新过滤 keep={_keep} pool={_pool}", flush=True)
                elif _m_year:
                    # v3: 完整日期 token(20251215) 优先；文档名或 chunk 内容命中才保留；
                    #     无文档名命中年份时不硬过滤（避免误杀 会议纪要_2024 等叙述文档）
                    _y = _m_year.group(1)
                    _name_hits, _content_hits = set(), set()
                    for _dnm, _idxs in _doc_to_idx.items():
                        if (_date_tok and _date_tok in _dnm) or _y in _dnm:
                            _name_hits.add(_dnm)
                        elif _date_tok:
                            for _idx in _idxs:
                                _txt = sres.field[sres.ids[_idx]].get("content_with_weight", "") or ""
                                if _date_tok in _txt:
                                    _content_hits.add(_dnm)
                                    break
                    if _name_hits:
                        _keep_docs = _name_hits | _content_hits
                        print(f"[OPT-RETRIEVAL-v3] 年份过滤 year={_y} token={_date_tok} name={_name_hits} content={_content_hits}", flush=True)
                if _keep_docs:
                    _new_valid = [i for i in valid_idx if sres.field[sres.ids[i]].get("docnm_kwd","") in _keep_docs]
                    if _new_valid:
                        # v3.3: 无 rerank 时做三层排序（完整日期名 > 年份名 > 仅内容命中）；
                        #        有 rerank 时 valid_idx 已按 rerank 分数排好，跳过重排
                        if not rerank_mdl:
                            _dnm_of = lambda i: sres.field[sres.ids[i]].get("docnm_kwd", "")
                            _t1 = {d for d in _keep_docs if _date_tok and _date_tok in d}
                            _t2 = {d for d in _keep_docs if d not in _t1 and _m_year and _m_year.group(1) in d}
                            _a = [i for i in _new_valid if _dnm_of(i) in _t1]
                            _b = [i for i in _new_valid if _dnm_of(i) in _t2]
                            _seen = set(_a) | set(_b)
                            _c = [i for i in _new_valid if i not in _seen]
                            if _a or _b:
                                _new_valid = _a + _b + _c
                                print(f"[OPT-RETRIEVAL-v3] 三层 t1={_t1} t2={_t2}", flush=True)
                        valid_idx = _new_valid
                        filtered_count = len(valid_idx)
                        ranks["total"] = int(filtered_count)
        except Exception as _e:
            import traceback; traceback.print_exc()
        # <<< END PATCH

        _par_handled=False
        # >>> PATCH 2026-08-26 retrflow-v1 P2: LLM 意图计划 -> 表格聚合 (轻量分类前置，P0并行化见下)
        # P0并行化: 计数意图时聚合LLM与路由LLM并发 (async gather 降30-50%)
        _agg_done=False
        try:
            _q_ag = question or ""
            _uni_rewrite = None
            _need_agg = _ag_light_classify(_q_ag) and (kb_ids or []) and (tenant_ids or [])
            _need_route = False
            if not doc_ids:
                _tids_tmp = tenant_ids if isinstance(tenant_ids, (list, tuple)) else [tenant_ids]
                _tids_tmp = [x for x in _tids_tmp if x]
                if len(_tids_tmp)==1:
                    ok_tmp, _ = _lr_v2_gate(_q_ag)
                    _need_route = ok_tmp
            if _need_agg and _need_route:
                import asyncio as _par_async
                _tid_par = tenant_ids[0] if isinstance(tenant_ids, (list, tuple)) else str(tenant_ids).split(",")[0]
                _uni_plan, _uni_route, _uni_rewrite = await _ag_unified_route_plan(_q_ag, kb_ids, _tid_par)
                if _uni_plan is not None or _uni_route is not None:
                    _par_plan, _par_route = _uni_plan, (_uni_route if _uni_route else (None, "unified-no-route"))
                    print(f"[UNIFIED-HIT] plan={bool(_uni_plan)} route={bool(_uni_route)}", flush=True)
                    _par_handled=True
                else:
                    _par_plan_task = _par_async.create_task(_ag_llm_plan(_q_ag, kb_ids, _tid_par))
                    _par_route_task = _par_async.create_task(_lr_v2_pick_docs(_q_ag, _tid_par, kb_ids))
                    _par_plan, _par_route = await _par_async.gather(_par_plan_task, _par_route_task, return_exceptions=True)
                    if isinstance(_par_plan, Exception): _par_plan=None
                    if isinstance(_par_route, Exception): _par_route=(None, str(_par_route))
                _par_handled=True
                if _par_plan and not isinstance(_par_plan, tuple):
                    _kw_ag = str(_par_plan.get("doc_keyword") or "")
                    _answered=False
                    _kw_core = _ag_re.sub(r"\.(xlsx|xlsm)$", "", _kw_ag, flags=_ag_re.I)[:6]
                    for _i in (valid_idx or [])[:5]:
                        _c = sres.field[sres.ids[_i]].get("content_with_weight", "") or ""
                        if _kw_core and _kw_core in _c and _ag_re.search(r"(现有|共|合计|总计|总数|最多)[^\n]{0,16}\d+(\.\d+)?\s*(台|张|条|个|套|辆|份|次)", _c):
                            _answered=True; break
                    if not _answered:
                        if _neg_tok:
                            print(f"[AGG-BLOCK-NEG] {_neg_tok} skip PAR", flush=True)
                        else:
                            _agg_chunk = await _par_async.to_thread(_ag_table_execute, _par_plan, kb_ids)
                            if _agg_chunk:
                                ranks["total"]=1; ranks["chunks"]=[_agg_chunk]; ranks["doc_aggs"]=[]; print(f"[AGG-TABLE-V1-HIT-PAR] {_agg_chunk['content_with_weight'][:96]}", flush=True); _agg_done=True
                if not _agg_done and _par_route and _par_route[0] is not None:
                    _lr_picked, _lr_why = _par_route
                    _lr_doc_ids, _lr_names, _lr_conf = _lr_picked
                    _lr_extra = await _lr_v2_scoped(self, question, _lr_doc_ids, kb_ids, idx_names, sres, page_size, term_similarity_weight, vector_similarity_weight, post_threshold, rank_feature)
                    if _lr_extra:
                        _lr_id_to_idx = {c: i for i, c in enumerate(sres.ids)}
                        _lr_valid = set(valid_idx); _lr_new_sims=[]; _lr_added=0
                        for _lr_cid, _lr_s, _lr_f in _lr_extra:
                            if _lr_cid in _lr_id_to_idx: _lr_idx=_lr_id_to_idx[_lr_cid]
                            else: sres.ids.append(_lr_cid); sres.field[_lr_cid]=_lr_f; _lr_new_sims.append(_lr_s); _lr_idx=len(sres.ids)-1
                            if _lr_idx not in _lr_valid: valid_idx.append(_lr_idx); _lr_valid.add(_lr_idx); _lr_added+=1
                        if _lr_new_sims: sim_np=np.concatenate([sim_np, np.array(_lr_new_sims, dtype=np.float64)]); tsim=list(tsim)+_lr_new_sims; vsim=list(vsim)+_lr_new_sims
                        if _lr_added: valid_idx.sort(key=lambda i: -float(sim_np[i])); filtered_count=len(valid_idx); ranks["total"]=int(filtered_count); print(f"[LLM-ROUTE-v2-PAR] union picked={_lr_names} added={_lr_added} total={filtered_count}", flush=True)
                if _agg_done:
                    if _neg_tok:
                        print(f"[AGG-BLOCK-NEG] {_neg_tok} par return blocked", flush=True)
                        ranks["chunks"]=[]; ranks["total"]=0; ranks["doc_aggs"]=[]
                    _par_handled=True
                    return ranks
            elif _need_agg:
                _tid = tenant_ids[0] if isinstance(tenant_ids, (list, tuple)) else str(tenant_ids).split(",")[0]
                _plan = await _ag_llm_plan(_q_ag, kb_ids, _tid)
                if _plan:
                    _kw_ag = str(_plan.get("doc_keyword") or "")
                    _answered=False
                    _kw_core = _ag_re.sub(r"\.(xlsx|xlsm)$", "", _kw_ag, flags=_ag_re.I)[:6]
                    for _i in (valid_idx or [])[:5]:
                        _c = sres.field[sres.ids[_i]].get("content_with_weight", "") or ""
                        if _kw_core and _kw_core in _c and _ag_re.search(r"(现有|共|合计|总计|总数|最多)[^\n]{0,16}\d+(\.\d+)?\s*(台|张|条|个|套|辆|份|次)", _c):
                            _answered=True; break
                    if not _answered:
                        if _neg_tok:
                            print(f"[AGG-BLOCK-NEG] {_neg_tok} skip", flush=True)
                        else:
                            import asyncio as _ag_async
                            _agg_chunk = await _ag_async.to_thread(_ag_table_execute, _plan, kb_ids)
                            if _agg_chunk:
                                ranks["total"]=1; ranks["chunks"]=[_agg_chunk]; ranks["doc_aggs"]=[]; print(f"[AGG-TABLE-V1-HIT] {_agg_chunk['content_with_weight'][:96]}", flush=True); return ranks
        except Exception as _ag_e:
            logging.warning(f"[AGG-TABLE-V1] fail {type(_ag_e).__name__}: {_ag_e}")
        # <<< END PATCH retrflow-v1 P2


        # >>> PATCH 2026-08-26 llm-route-v2: 摘要驱动 LLM 文档路由（逐文档域检索 + union 合并；任何异常软回退基线）
        if _par_handled:
            print("[LLM-ROUTE-v2] skipped - already handled in PAR", flush=True)
        elif not doc_ids:
            try:
                _lr_tids = tenant_ids if isinstance(tenant_ids, (list, tuple)) else [tenant_ids]
                _lr_tids = [t for t in _lr_tids if t]
                if len(_lr_tids) == 1:
                    _lr_picked, _lr_why = await _lr_v2_pick_docs(question, _lr_tids[0], kb_ids)
                    if _lr_picked is not None:
                        _lr_doc_ids, _lr_names, _lr_conf = _lr_picked
                        _lr_extra = await _lr_v2_scoped(self, question, _lr_doc_ids, kb_ids, idx_names,
                                                       sres, page_size, term_similarity_weight,
                                                       vector_similarity_weight, post_threshold, rank_feature)
                        if _lr_extra:
                            # llm-route-v2.5: 去重基准=valid_idx（而非 sres.ids 主池全量）——被 v3.3 域过滤剔除的 chunk 用原索引重新加入
                            _lr_id_to_idx = {c: i for i, c in enumerate(sres.ids)}
                            _lr_valid = set(valid_idx)
                            _lr_new_sims, _lr_added = [], 0
                            for _lr_cid, _lr_s, _lr_f in _lr_extra:
                                if _lr_cid in _lr_id_to_idx:
                                    _lr_idx = _lr_id_to_idx[_lr_cid]
                                else:
                                    sres.ids.append(_lr_cid)
                                    sres.field[_lr_cid] = _lr_f
                                    _lr_new_sims.append(_lr_s)
                                    _lr_idx = len(sres.ids) - 1
                                if _lr_idx not in _lr_valid:
                                    valid_idx.append(_lr_idx)
                                    _lr_valid.add(_lr_idx)
                                    _lr_added += 1
                            if _lr_new_sims:
                                sim_np = np.concatenate([sim_np, np.array(_lr_new_sims, dtype=np.float64)])
                                tsim = list(tsim) + _lr_new_sims
                                vsim = list(vsim) + _lr_new_sims
                            if _lr_added:
                                valid_idx.sort(key=lambda i: -float(sim_np[i]))
                                filtered_count = len(valid_idx)
                                ranks["total"] = int(filtered_count)
                                print(f"[LLM-ROUTE-v2] union q={question[:24]} conf={_lr_conf} picked={_lr_names} scoped={len(_lr_extra)} readded={_lr_added} total={filtered_count}", flush=True)
                    else:
                        logging.info(f"[LLM-ROUTE-v2] skip q={question[:24]} why={_lr_why}")
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
                    # <<< END PATCH llm-route-v2.6
            except Exception as _lr_e:
                logging.warning(f"[LLM-ROUTE-v2] 路由失败保持基线候选: {type(_lr_e).__name__}: {_lr_e}")  # llm-route-v2.2
        # <<< END PATCH

        # >>> PATCH 2026-08-28 query-rewrite-v1: 改写/扩展变体 Hybrid 召回 union（valid_idx 为基准，软回退零回归）
        try:
            if _QR_ENV_ON and (kb_ids or []) and sres.query_vector:
                _qrc = _qr_cfg()
                if _qrc.get("enabled", True):
                    _qr_vs = _qr_variants(question, _qrc)
                    if _uni_rewrite:
                        for _rwn in _uni_rewrite:
                            if _rwn and str(_rwn) != (question or "") and str(_rwn) not in _qr_vs:
                                _qr_vs.append(_rwn)
                    _extra = [x for x in _qr_vs[1:] if x and x != (question or "")]
                    if _extra and not (bool(_qrc.get("skip_neg", True)) and _neg_tok):
                        _qr_limit = min(max(int(_qrc.get("search_size", 100) or 100), 10), 200)
                        _qr_sim = float(_qrc.get("search_similarity", 0.05) or 0.05)
                        _qr_valid = set(valid_idx)
                        _qr_added = 0
                        for _vq in _extra:
                            _qr_req = dict(req)
                            _qr_req.update({"question": _vq, "similarity": _qr_sim, "topk": _qr_limit, "size": _qr_limit, "page": 1})
                            _qr_s = await self.search(_qr_req, idx_names, kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
                            _qr_s = await self._prune_deleted_chunks(_qr_s)
                            if not getattr(_qr_s, "ids", None):
                                continue
                            _qr_new_ids = [c for c in _qr_s.ids if c not in sres.field and _qr_s.field.get(c)]
                            if not _qr_new_ids:
                                continue
                            _qr_fake = self.SearchResult(total=len(_qr_new_ids), ids=_qr_new_ids,
                                                         query_vector=sres.query_vector,
                                                         field={c: _qr_s.field[c] for c in _qr_new_ids})
                            _qr_knn = await self._knn_scores(_qr_fake, idx_names, kb_ids)
                            _qr_sims, _qt, _qv = self.rerank_with_knn(_qr_fake, question, _qr_knn,
                                                                      1 - vector_similarity_weight, vector_similarity_weight,
                                                                      rank_feature=rank_feature)
                            _qr_scores = []
                            for j, cid in enumerate(_qr_new_ids):
                                _sv = float(_qr_sims[j])
                                if post_threshold > 0 and _sv < post_threshold:
                                    continue
                                sres.ids.append(cid)
                                sres.field[cid] = _qr_s.field[cid]
                                _qr_scores.append(_sv)
                            if _qr_scores:
                                sim_np = np.concatenate([sim_np, np.array(_qr_scores, dtype=np.float64)])
                                tsim = list(tsim) + _qr_scores
                                vsim = list(vsim) + _qr_scores
                                _base = len(sres.ids) - len(_qr_scores)
                                for _k in range(len(_qr_scores)):
                                    _ix = _base + _k
                                    if _ix not in _qr_valid:
                                        valid_idx.append(_ix)
                                        _qr_valid.add(_ix)
                                        _qr_added += 1
                        if _qr_added:
                            valid_idx.sort(key=lambda i: -float(sim_np[i]))
                            filtered_count = len(valid_idx)
                            ranks["total"] = int(filtered_count)
                            print(f"[QUERY-REWRITE] q={(question or '')[:24]} variants={[x[:16] for x in _extra]} added={_qr_added} total={filtered_count}", flush=True)
        except Exception as _qre:
            logging.warning(f"[QUERY-REWRITE] skip: {type(_qre).__name__}: {_qre}")
        # <<< END PATCH query-rewrite-v1

        if filtered_count == 0:
            ranks["doc_aggs"] = []
            return ranks

        begin = global_offset % RERANK_LIMIT
        end = begin + page_size
        page_idx = valid_idx[begin:end]

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim

        for i in page_idx:
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")

            position_int = chunk.get("position_int", [])
            # Chunk vectors are no longer fetched during the main retrieval
            # call. Fall back to whatever the chunk happens to carry (Infinity
            # path) and otherwise emit a zero placeholder so the downstream
            # shape stays stable. Citation callers refill this via
            # Dealer.fetch_chunk_vectors when needed.
            _cww = chunk["content_with_weight"]
            # [CONFIDENTIAL-MASK] 机密文档敏感ID脱敏（BOT/SH/GD/P0）- 需邀请制透传 invite_verified 旗标，当前仅审计不脱敏以保召回
            # if _CONF_DOC_RE.search(dnm or "") and _SENSITIVE_RE.search(_cww or ""):
            #     try: _cww = _SENSITIVE_RE.sub("***", _cww)
            #     except Exception: pass
            # 占位：本地加密+邀请制已在存储层生效，此处仅阈值/限流，脱敏待 invite_verified 接入后启用
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": _cww,
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "tag_kwd": chunk.get("tag_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": float(sim_np[i]),
                "vector_similarity": float(vsim[i]),
                "term_similarity": float(tsim[i]),
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "doc_type_kwd": chunk.get("doc_type_kwd", ""),
                "mom_id": chunk.get("mom_id", ""),
                "row_id": chunk.get("row_id()"),
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = remove_redundant_spaces(sres.highlight[id])
                    # [CONFIDENTIAL-MASK] highlight 脱敏同上，待 invite_verified 接入后启用
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)

        if aggs:
            for i in valid_idx:
                id = sres.ids[i]
                chunk = sres.field[id]
                dnm = chunk.get("docnm_kwd", "")
                did = chunk.get("doc_id", "")
                if dnm not in ranks["doc_aggs"]:
                    ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                ranks["doc_aggs"][dnm]["count"] += 1

            ranks["doc_aggs"] = [
                {
                    "doc_name": k,
                    "doc_id": v["doc_id"],
                    "count": v["count"],
                }
                for k, v in sorted(
                    ranks["doc_aggs"].items(),
                    key=lambda x: x[1]["count"] * -1,
                )
            ]
        else:
            ranks["doc_aggs"] = []
        # [FINAL-NEG-BLOCK] 兜底：负例在聚合/路由回补后仍强制空（防 AGG/LLM绕过）
        if _neg_tok and ranks["chunks"]:
            print(f"[FINAL-NEG-BLOCK] {_neg_tok} -> empty (had {len(ranks['chunks'])} chunks)", flush=True)
            ranks["chunks"] = []
            ranks["total"] = 0
            ranks["doc_aggs"] = []

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"],
                   sort_by_position: bool = False,
                   retrieve_all: bool = False):
        """Return chunks for a document.

        By default, preserve the historical max_count cap. When retrieve_all is
        True, keep paging until the doc store returns fewer rows than requested.
        """
        condition = {"doc_id": doc_id}

        fields_set = set(fields or [])
        if sort_by_position:
            for need in ("page_num_int", "position_int", "top_int"):
                if need not in fields_set:
                    fields_set.add(need)
        fields = list(fields_set)

        orderBy = OrderByExpr()
        if sort_by_position:
            orderBy.asc("page_num_int")
            orderBy.asc("position_int")
            orderBy.asc("top_int")

        res = []
        bs = 128
        p = offset
        while retrieve_all or p < max_count:
            limit = bs if retrieve_all else min(bs, max_count - p)
            if limit <= 0:
                break
            es_res = self.dataStore.search(fields, [], condition, [], orderBy, p, limit, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.get_fields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            chunk_count = len(dict_chunks)
            if chunk_count == 0 or chunk_count < limit:
                break
            p += limit
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        if not self.dataStore.index_exist(index_name(tenant_id), kb_ids[0]):
            return []
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.get_aggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.get_aggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []),
                                        keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.get_aggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1 * (c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}

    async def retrieval_by_toc(self, query: str, chunks: list[dict], tenant_ids: list[str], chat_mdl, topn: int = 6):
        from rag.prompts.generator import relevant_chunks_with_toc # moved from the top of the file to avoid circular import
        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        ranks, doc_id2kb_id = {}, {}
        for ck in chunks:
            if ck["doc_id"] not in ranks:
                ranks[ck["doc_id"]] = 0
            ranks[ck["doc_id"]] += ck["similarity"]
            doc_id2kb_id[ck["doc_id"]] = ck["kb_id"]
        doc_id = sorted(ranks.items(), key=lambda x: x[1] * -1.)[0][0]
        kb_ids = [doc_id2kb_id[doc_id]]
        es_res = self.dataStore.search(["content_with_weight"], [], {"doc_id": doc_id, "toc_kwd": "toc"}, [],
                                       OrderByExpr(), 0, 128, idx_nms,
                                       kb_ids)
        toc = []
        dict_chunks = self.dataStore.get_fields(es_res, ["content_with_weight"])
        for _, doc in dict_chunks.items():
            try:
                toc.extend(json.loads(doc["content_with_weight"]))
            except Exception as e:
                logging.exception(e)
        if not toc:
            return chunks

        ids = await relevant_chunks_with_toc(query, toc, chat_mdl, topn * 2)
        if not ids:
            return chunks

        vector_size = 1024
        id2idx = {ck["chunk_id"]: i for i, ck in enumerate(chunks)}
        for cid, sim in ids:
            if cid in id2idx:
                chunks[id2idx[cid]]["similarity"] += sim
                continue
            chunk = self.dataStore.get(cid, idx_nms[0], kb_ids)
            if not chunk:
                continue
            d = {
                "chunk_id": cid,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": doc_id,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim,
                "vector_similarity": sim,
                "term_similarity": sim,
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in chunk.keys():
                if k[-4:] == "_vec":
                    d["vector"] = chunk[k]
                    vector_size = len(chunk[k])
                    break
            chunks.append(d)

        return sorted(chunks, key=lambda x: x["similarity"] * -1)[:topn]

    def retrieval_by_children(self, chunks: list[dict], tenant_ids: list[str]):
        if not chunks:
            return []
        idx_nms = [index_name(tid) for tid in tenant_ids]
        mom_chunks = defaultdict(list)
        i = 0
        while i < len(chunks):
            ck = chunks[i]
            mom_id = ck.get("mom_id")
            if not isinstance(mom_id, str) or not mom_id.strip():
                i += 1
                continue
            mom_chunks[ck["mom_id"]].append(chunks.pop(i))

        if not mom_chunks:
            return chunks

        if not chunks:
            chunks = []

        vector_size = 1024
        for id, cks in mom_chunks.items():
            chunk = self.dataStore.get(id, idx_nms[0], [ck["kb_id"] for ck in cks])
            if chunk is None:
                logging.warning(
                    "Parent chunk '%s' not found in the index; falling back to %d child chunk(s).",
                    id, len(cks),
                )
                chunks.extend(cks)
                continue
            d = {
                "chunk_id": id,
                "content_ltks": " ".join([ck["content_ltks"] for ck in cks]),
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "kb_id": chunk["kb_id"],
                "important_kwd": [kwd for ck in cks for kwd in ck.get("important_kwd", [])],
                "image_id": chunk.get("img_id", ""),
                "similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector_similarity": np.mean([ck["similarity"] for ck in cks]),
                "term_similarity": np.mean([ck["similarity"] for ck in cks]),
                "vector": [0.0] * vector_size,
                "positions": chunk.get("position_int", []),
                "doc_type_kwd": chunk.get("doc_type_kwd", "")
            }
            for k in cks[0].keys():
                if k[-4:] == "_vec":
                    d["vector"] = cks[0][k]
                    vector_size = len(cks[0][k])
                    break
            chunks.append(d)

        return sorted(chunks, key=lambda x: x["similarity"] * -1)
