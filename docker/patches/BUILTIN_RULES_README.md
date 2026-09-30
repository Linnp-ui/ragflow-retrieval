# 检索硬编码剥离 (builtin-rules-extract) — 2026-08-28

## 目标

把 search.py 中三块爱博产品线硬编码搬到 `/ragflow/conf/builtin_retrieval_rules.json`：

1. **域路由** `_domain_rules` (v3 块，6 条爱博产品名匹配)
2. **负例正则** `_NEG_RE` (retrflow-v3，P0[4-9]/P09/GD999/... 爱博化)
3. **LLM PROMPT 例子** (v2 PROMPT，`P01样机表__20240410更新` 爱博命名)

## 设计原则

- **零回归**：所有 3 块内置默认 = 原硬编码；配置缺失/解析失败 → fallback 默认
- **env 总开关**：`RAGFLOW_BUILTIN_RULES=0` → 整套规则走默认（他人租户开箱即用纯基线）
- **配置热加载**：30s mtime 轮询，无需重启
- **tenant scope**：`_apply_tenant_ids` 限定规则生效的租户（空 → 全部）
- **关闭某段**：配置对应字段留空（`patterns: []` / `domain_rules: []`）→ 该段失效，**其他段不受影响**
- **幂等**：4 个补丁独立 MARK 守卫，可重复运行

## 文件清单

| 文件 | 作用 |
|---|---|
| `builtin_retrieval_rules.json` | 配置文件（爱博基线 + tenant scope `_apply_tenant_ids: null`） |
| `patch_builtin_rules_loader.py` | 注入 `_brl_load_cfg/_brl_get_domain_rules/_brl_get_neg_patterns/_brl_get_llm_route_examples` 模块级函数 |
| `patch_retrieval_rules.py` | 替换 v3 `_domain_rules` 块，调用 `_brl_get_domain_rules` |
| `patch_neg_rules.py` | 替换 `_NEG_RE = re.compile(...)` 行为，调用 `_brl_get_neg_patterns` |
| `patch_llm_route_prompt.py` | 替换 v2 PROMPT 里的 P0x 例子，调用 `_brl_get_llm_route_examples` |
| `apply_builtin_rules.sh` | 顺序运行 4 个补丁 + 配置同步提示 |

## 应用步骤

```bash
cd ~/RagSystem/ragflow/docker/patches
bash apply_builtin_rules.sh
# 同步配置 (scp 后):
sudo mkdir -p /ragflow/conf
sudo cp builtin_retrieval_rules.json /ragflow/conf/
sudo chmod 644 /ragflow/conf/builtin_retrieval_rules.json

# 重载:
cd /home/abrobo/RagSystem/ragflow && docker compose up -d ragflow-cpu

# 验证:
docker exec docker-ragflow-cpu-1 python3 -c \
  "import rag.nlp.search as s; print('BRL OK' if hasattr(s, '_brl_get_domain_rules') else 'MISSING')"
```

## 配置示例 (他人租户)

要让爱博基线**对其他租户不生效**，二选一：

**A. env 总开关 (推荐)**
```bash
# /home/abrobo/RagSystem/ragflow/.env
RAGFLOW_BUILTIN_RULES=0
```

**B. tenant scope 限定 (更精细，可保留爱博租户的规则)**
```json
{
  "_apply_tenant_ids": ["49096a1e"],
  ...
}
```
其他租户请求触发 `_brl_get_*` 时返回 `None` → fallback 默认 → 默认也是爱博硬编码列表，等同爱博零回归但其他租户也是爱博规则。
**注意：tenant scope 仅控制配置是否加载；默认 fallback 仍是爱博硬编码。**
要让他人租户**完全跳过**该段，必须同时：
1. 设 `_apply_tenant_ids` 限定爱博
2. **且** 默认列表本身改成空（不推荐，会影响爱博）

**最简方案**：所有租户统一走 A 方案 (env disable) + 爱博租户在 .env **不设** RAGFLOW_BUILTIN_RULES，配置文件加载爱博规则。

## 风险与验证

### 零回归验证 (改后必跑)
- 121 题 aibao_eval_set：`H@1 ≥ 0.95` (目标与改前 0.963 同档 ± 0.02)
- L2 20/20 + NEG 14/14 (全阈值不变)

### 行为变化 (改后必看)
- 检索日志出现 `[BRL] domain_rules from config: 6 entries (tenant=...)` → 走配置
- 不出现 → 走内置默认 (爱博零回归)

### 副作用检查
- 改 `builtin_retrieval_rules.json` 后**无需** docker restart（30s 热加载）
- 改 search.py 4 个补丁后**必须** `docker compose up -d`
- 4 补丁的 `bak-*-20260828` 备份在同目录，便于回退

## 兜底回退

```bash
# 整套回退到改前 (search.py):
ls -t /home/abrobo/RagSystem/ragflow/docker/patches/search.py.bak-*-20260828 | head -1 | \
  xargs -I{} sudo cp {} /home/abrobo/RagSystem/ragflow/docker/patches/search.py
cd /home/abrobo/RagSystem/ragflow && docker compose up -d ragflow-cpu
```

## 已知坑

1. **patch_neg_rules.py 用 _NEG_RE 字面量定位** —— 若 search.py 重构时改了变量名或字符串拼接方式，需重写锚点
2. **patch_llm_route_prompt.py 替换了 `.format(...)` 调用** —— 若 v2 helper 重构，需重写
3. **tenant_ids 在 retrieval() 内是 list/tuple** —— 三个 `_brl_get_*` 都做了 None 兜底，但若未来加 `str` 类型以外的 tenant 形态，需在 `_brl_tenant_allowed` 扩展
4. **PYTHONHASHSEED** —— 配置热加载用 dict,key 顺序在 Python 3.7+ 保证；不影响行为
5. **多 worker** —— `_BRL_STATE` 是模块级 dict,各 worker 独立缓存（30s 内各拉一次，无一致性问题）

## 未来扩展

- 把 `_BUILTIN_DOMAIN_RULES_DEFAULT` 移到 `/ragflow/conf/_builtin_defaults/domain_rules.json`，连默认都搬空（彻底无爱博痕迹）
- 增加 `_brl_log` 统计段：每个规则的命中率/误伤率，用于 A/B
- 把 llm_route_examples 扩展到 `_brl_get_complex_rules` (COMPLEX_RULES 也搬出)
