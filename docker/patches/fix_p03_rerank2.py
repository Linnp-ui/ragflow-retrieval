import pathlib
p=pathlib.Path("/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
t=p.read_text(encoding="utf-8")
# remove old rerank patch and replace with stronger one
if "P03-RERANK" in t:
    # find old block
    import re
    t=re.sub(r"# >>> PATCH 2026-08-28 ab57-p03-rerank:.*?print\(f\"\[P03-RERANK-ERR\].*?\)", "# >>> PATCH 2026-08-28 ab57-p03-rerank-v2: P03强制首位\n        try:\n            if \"P03\" in (question or \"\"):\n                _all_ids = list(sorted_idx)\n                _p03 = [int(i) for i in _all_ids if \"P03\" in str(sres.field[sres.ids[int(i)]].get(\"docnm_kwd\",\"\") or \"\")]\n                _oth = [int(i) for i in _all_ids if int(i) not in _p03]\n                if _p03:\n                    sorted_idx = __import__(\"numpy\").array(_p03 + _oth)\n                    open(\"/tmp/p03_boost.log\",\"a\").write(f\"boost {question[:12]} {len(_p03)}/{len(_all_ids)}\\n\")\n                    print(f\"[P03-RERANK-V2] boosted {len(_p03)} P03 to front\", flush=True)\n        except Exception as _e:\n            open(\"/tmp/p03_err.log\",\"a\").write(str(_e)+\"\\n\")\n            print(f\"[P03-RERANK-ERR2] {_e}\", flush=True)", t, flags=re.S)
    p.write_text(t,encoding="utf-8")
    print("replaced with v2")
else:
    print("not found")
