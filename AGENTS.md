# AGENTS.md

Verified against the working tree. Deep-dive doc: `CLAUDE.md` (Python architecture).

## Scope: Python backend

The parallel Go/C++ backend rewrite (`internal/`, `cmd/`, `build.sh`, `internal/cpp`) was removed
in 72a7752 — it was never running in production. There is now one backend: Python. Don't add new
infrastructure expecting a second implementation, and don't edit `web/vite.config.ts`
`proxySchemes` for new routes (the `python` scheme already forwards all of `/api` and `/v1` to
9380).

## Editing code: thirteen files live in `docker/patches/`

`docker/docker-compose.yml` bind-mounts `docker/patches/*.py` over 13 main-tree modules
(`search.py`, `mcp_server.py`, `provider_api.py`, `embedding_model.py`, `dialog_service.py`,
`api_apps_init.py`, …). **The patch copy is what executes.** Editing the main-tree file has no
effect at all. README.md has the full mapping table.

After editing anything under `docker/patches/`, restart and verify:

```bash
cd docker && docker compose -f docker-compose.yml restart ragflow-cpu   # up -d will NOT restart
md5sum patches/<file>.py && docker exec docker-ragflow-cpu-1 md5sum /ragflow/<mounted path>
```

`restart` is mandatory, not optional: a bind mount binds an inode, and any tool that saves by
rename (`sed -i`, most editors) leaves the mount pointing at the old file. The container then serves
stale code with no error — and possibly a half-updated file.

## Setup

```bash
uv sync --python 3.13 --all-extras   # Python is pinned to >=3.13,<3.14
uv run python3 download_deps.py     # required: fetches nltk_data used by tokenizers
pre-commit install
docker compose -f docker/docker-compose-base.yml up -d   # MySQL/ES/Redis/MinIO
```

`download_deps.py` is not optional. `test/unit_test/conftest.py` reuses `./nltk_data`, and without it
tokenizer-backed tests fail with `LookupError: Resource 'punkt_tab' not found`.

## Running

```bash
export PYTHONPATH=$(pwd)
bash docker/launch_backend_service.sh              # ragflow + task_executor
bash docker/launch_backend_service.sh task_executor # or: ragflow | admin | data_sync

cd web && export API_PROXY_SCHEME=python && npm run dev   # Vite, port 9222
```

`API_PROXY_SCHEME` should be `python`. `web/vite.config.ts` still carries `hybrid`/`go` proxy
tables, but both are dead — they target ports 9384/9383, which no longer have a listener.

## Tests

```bash
uv sync --python 3.13 --group test   # pytest & co live in the `test` group, not --all-extras
uv pip install -e sdk/python         # required for test/testcases/test_sdk_api

uv run pytest test/unit_test        # pure unit tests — the only suite that runs without a stack
uv run pytest test/unit_test/test_x.py -k name
python run_tests.py -i              # what CI runs: unit_test only, -i ignores SyntaxWarning
python run_tests.py -p -c -t path -k kw -m p1   # parallel / coverage / filter
```

Quirks:
- **Run the whole `test/unit_test`, not single files.** `filterwarnings = ["error", ...]` plus
  import-time deprecation warnings means a file that passes in a full run can fail collection when
  run alone (`test_dataflow_service.py` raises `UserWarning: local_dir_use_symlinks`). Warnings
  raised at import are only emitted once per process.
- Bare `uv run pytest` collects `test/testcases/` (needs a live server via `HOST_ADDRESS`) and
  `test/playwright/` (needs browsers). Scope to `test/unit_test` for quick checks.
- `scholarly==1.7.11` (pinned in `uv.lock`) is not Python 3.13 compatible — `re.search("cites=[\d+…")`
  without the `r` prefix is a hard `SyntaxError` there, which breaks collection of any test importing
  `agent/tools/googlescholar.py`. Patch line 312 of `.venv/.../scholarly/_scholarly.py` locally, or
  `--ignore` the affected files. Upstream is affected too.
- `test/testcases/` takes a custom `--level p1|p2|p3` flag (defined in its `conftest.py`) that
  rewrites `markexpr`; markers `p0`–`p3`, `smoke`, `auth`, `asyncio` are declared in
  `pyproject.toml`. CI uses `p2` for PRs and `p3` nightly.
- `filterwarnings = ["error", ...]` — warnings fail tests. Don't "fix" a test by suppressing a
  warning you introduced.
- `asyncio_mode = "auto"`: async test functions need no decorator.
- Files named `test_*_routes_unit.py` under `test/testcases/restful_api/` are the offline route
  tests; the siblings without `_unit` hit the running server.
- Each API test suite runs twice in CI against `DOC_ENGINE=infinity` then `elasticsearch`; both
  doc-store code paths must work.
- `download_deps.py` also fetches chromedriver, tika and two HuggingFace models that
  `test/unit_test` never loads — `nltk.download("wordnet"/"punkt"/"punkt_tab")` is the only part
  needed, and the zips may need extracting by hand.
- CI runs `ruff check` → `run_tests.py -i`. Nothing else gates Python changes.

## Lint / typecheck

```bash
ruff check && ruff format          # line-length is 200, not 88; E402 ignored, ASYNC/ASYNC1 enabled
cd web && npm run lint             # eslint
cd web && npm run type-check       # tsc --noEmit
cd web && npm run test             # jest --coverage
```

## Conventions

- `check-yaml`, `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending` etc. run via
  pre-commit on every commit.
- Python: async everywhere (`Quart`, not Flask). Components talk through the canvas graph in
  `agent/canvas.py` / `rag/flow/pipeline.py`; variable refs look like `{component_id@output_var}`.
- `DOC_ENGINE` in `docker/.env` (`elasticsearch` | `infinity` | `opensearch` | `oceanbase`) is
  read at process start — change it and restart, don't expect a live switch.
- `docker/.env` holds all deployment env; `docker/service_conf.yaml.template` is the backend config
  template.
- `check_comment_ascii.py` exists but its CI step is disabled (`if: ${{ false }}`).
