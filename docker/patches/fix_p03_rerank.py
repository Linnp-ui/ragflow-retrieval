import pathlib
p=pathlib.Path("/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
t=p.read_text(encoding="utf-8")
anchor="        sorted_idx = np.argsort(sim_np * -1, kind='stable')"
if "P03-RERANK" in t:
    print("already")
else:
    patch=anchor + """
        # >>> PATCH 2026-08-28 ab57-p03-rerank: P03/P02 消歧后排升权
        try:
            if "P03" in (question or ""):
                _p03 = [int(i) for i in sorted_idx if "P03" in str(sres.field[sres.ids[int(i)]].get("docnm_kwd","") or "") or "NRV" in str(sres.field[sres.ids[int(i)]].get("content_with_weight","") or "")]
                _oth = [int(i) for i in sorted_idx if int(i) not in _p03]
                if _p03 and _oth and float(sim_np[_p03[0]]) + 0.05 >= float(sim_np[_oth[0]]):
                    sorted_idx = __import__("numpy").array(_p03 + _oth, dtype=__import__("numpy").int64)
                    print(f"[P03-RERANK] boosted q={(question or '')[:18]} top {sim_np[_oth[0]]:.4f}->{sim_np[_p03[0]]:.4f}", flush=True)
        except Exception as _e:
            print(f"[P03-RERANK-ERR] {_e}", flush=True)"""
    t=t.replace(anchor, patch)
    p.write_text(t,encoding="utf-8")
    print("patched rerank")
