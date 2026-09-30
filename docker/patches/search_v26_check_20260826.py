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

        valid_idx = [int(i) for i in sorted_idx if sim_np[i] >= post_threshold]
        filtered_count = len(valid_idx)
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
            _domain_rules = [
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
            # --- P3: 路由类查询限制返回条数，防大响应压垮 worker ---
            if (_is_latest or _m_year or _matched_domain_docs is not None) and page_size > 20:
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

        # >>> PATCH 2026-08-26 llm-route-v2: 摘要驱动 LLM 文档路由（逐文档域检索 + union 合并；任何异常软回退基线）
        if not doc_ids:
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
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
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
