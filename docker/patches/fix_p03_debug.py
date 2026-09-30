import pathlib
p=pathlib.Path("/home/abrobo/RagSystem/ragflow/docker/patches/search.py")
t=p.read_text(encoding="utf-8")
if "P03-DEBUG" not in t:
    t=t.replace(
        "_orig_w = vector_similarity_weight",
        "print(f\"[P03-DEBUG] q={(question or chr(39))[:12]} w={vector_similarity_weight}\", flush=True)\n        _orig_w = vector_similarity_weight"
    )
    p.write_text(t,encoding="utf-8")
    print("added debug")
else:
    print("already")
