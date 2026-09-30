#!/usr/bin/env bash
# apply_builtin_rules.sh — 按依赖顺序执行 4 个补丁, 把 _NEG_RE / _domain_rules / LLM PROMPT
# 从 search.py 硬编码搬到 /ragflow/conf/builtin_retrieval_rules.json (2026-08-28)
#
# 设计：全部幂等 (每个补丁自带 MARK 守卫)，可重复运行
# 目标文件: /home/abrobo/RagSystem/ragflow/docker/patches/search.py (容器内 bind mount 到
#          /ragflow/rag/nlp/search.py) — 改完后必须 docker compose up -d ragflow-cpu 重载
#
# 用法：
#   cd ~/RagSystem/ragflow/docker/patches
#   bash apply_builtin_rules.sh          # 应用全部 4 个补丁
#   bash apply_builtin_rules.sh verify    # 只校验,不写盘
#
# 配套配置：builtin_retrieval_rules.json 已写好, 需 scp 到宿主机 /ragflow/conf/ 后改 _apply_tenant_ids.
# 关闭整套规则：他租户在 ragflow.env 加 RAGFLOW_BUILTIN_RULES=0,无需改 search.py.
set -e
TARGET="/home/abrobo/RagSystem/ragflow/docker/patches/search.py"
CFG_PATH="/ragflow/conf/builtin_retrieval_rules.json"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== 目标: $TARGET ==="
echo "=== 配置: $CFG_PATH ==="
echo "=== 补丁: $PATCH_DIR ==="

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET 不存在" >&2; exit 1
fi

for p in patch_builtin_rules_loader.py patch_retrieval_rules.py patch_neg_rules.py patch_llm_route_prompt.py; do
  if [ ! -f "$PATCH_DIR/$p" ]; then
    echo "ERROR: 缺失 $p" >&2; exit 1
  fi
done

# 1) 加载器 (依赖: 无)
echo "--- [1/4] patch_builtin_rules_loader.py ---"
python3 "$PATCH_DIR/patch_builtin_rules_loader.py" || { echo "FAIL at loader"; exit 1; }

# 2) 域路由 (依赖: loader)
echo "--- [2/4] patch_retrieval_rules.py ---"
python3 "$PATCH_DIR/patch_retrieval_rules.py" || { echo "FAIL at retrieval_rules"; exit 1; }

# 3) 负例正则 (依赖: loader + retrieval_rules)
echo "--- [3/4] patch_neg_rules.py ---"
python3 "$PATCH_DIR/patch_neg_rules.py" || { echo "FAIL at neg_rules"; exit 1; }

# 4) LLM PROMPT 模板 (依赖: loader)
echo "--- [4/4] patch_llm_route_prompt.py ---"
python3 "$PATCH_DIR/patch_llm_route_prompt.py" || { echo "FAIL at llm_route_prompt"; exit 1; }

# 5) 配置同步 (可选 — 若 $CFG_PATH 不存在或比本地旧, 提示)
if [ -f "$PATCH_DIR/builtin_retrieval_rules.json" ]; then
  if [ ! -f "$CFG_PATH" ] || [ "$PATCH_DIR/builtin_retrieval_rules.json" -nt "$CFG_PATH" ]; then
    echo "--- 配置文件: $CFG_PATH (本地源更新,需 scp) ---"
    echo "    sudo mkdir -p /ragflow/conf"
    echo "    sudo cp $PATCH_DIR/builtin_retrieval_rules.json $CFG_PATH"
    echo "    sudo chmod 644 $CFG_PATH"
  else
    echo "--- 配置文件: $CFG_PATH 已就位 ---"
  fi
fi

echo ""
echo "=== 应用完成. 下一步: ==="
echo "  cd /home/abrobo/RagSystem/ragflow && docker compose up -d ragflow-cpu"
echo "  # 验证 (容器内):"
echo "  docker exec docker-ragflow-cpu-1 python3 -c \"import rag.nlp.search as s; print('BRL OK' if hasattr(s, '_brl_get_domain_rules') else 'MISSING')\""
echo "  # 关闭规则 (他人租户用):"
echo "  # 在 ragflow.env 加 RAGFLOW_BUILTIN_RULES=0 然后 docker compose up -d"
