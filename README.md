# RAGFlow Retrieval

A personal project: a customised fork of
[RAGFlow](https://github.com/infiniflow/ragflow) v0.26.0, carrying a retrieval-tuning layer and a
narrower permission model. Built for a specific deployment, published as-is — see
[Licensing](#licensing) before reusing it.

> [!IMPORTANT]
> This is **not** the upstream RAGFlow project, and it is not affiliated with or endorsed by
> InfiniFlow.
>
> **This fork does not track upstream.** The parallel Go/C++ backend that upstream carried was
> removed here (commit `72a7752`), so a rebase against upstream is not a mechanical operation. Use
> upstream for reference, not as a merge source.

## Read this before editing code

**Thirteen files in the main tree are bind-mounted over at runtime.** `docker/docker-compose.yml`
mounts `docker/patches/*.py` on top of the real module paths, so the patch copy is what executes.
Editing the main-tree file has **no effect** — the change is silently ignored.

| Patch (edit this) | Mounted over |
|---|---|
| `docker/patches/api_apps_init.py` | `api/apps/__init__.py` |
| `docker/patches/dataset_api_service.py` | `api/apps/services/dataset_api_service.py` |
| `docker/patches/dialog_service.py` | `api/db/services/dialog_service.py` |
| `docker/patches/embedding_model.py` | `rag/llm/embedding_model.py` |
| `docker/patches/figure_parser.py` | `deepdoc/parser/figure_parser.py` |
| `docker/patches/file_service.py` | `api/db/services/file_service.py` |
| `docker/patches/knowledgebase_service.py` | `api/db/services/knowledgebase_service.py` |
| `docker/patches/mcp_server.py` | `mcp/server/server.py` |
| `docker/patches/provider_api.py` | `api/apps/restful_apis/provider_api.py` |
| `docker/patches/search.py` | `rag/nlp/search.py` |
| `docker/patches/table.py` | `rag/app/table.py` |
| `docker/patches/task_service.py` | `api/db/services/task_service.py` |
| `docker/patches/tenant_model_instance_service.py` | `api/db/services/tenant_model_instance_service.py` |

This already cost one change: the MCP `vector_similarity_weight` default of `0.6` was applied to
`mcp/server/server.py` and did nothing, because `docker/patches/mcp_server.py` still said `0.3`.
The fix belongs in the patch (commit `e31a07a`).

Verify what is actually mounted rather than trusting the file list:

```bash
cd docker && docker compose config | grep patches/
```

After changing a mounted file, restart the container — the mount is read at process start.

## Restarting after a patch change

```bash
cd docker
docker compose -f docker-compose.yml restart ragflow-cpu
```

Two traps, both hit in practice:

**`up -d` does not restart.** It prints `Container ... Running` and returns, because the compose
configuration is unchanged — only the file *inside* the mount changed. Use `restart`, or
`up -d --force-recreate` if the mount list itself changed.

**A bind mount binds an inode, not a path.** Editors and `sed -i` typically save by writing a
temporary file and renaming it over the original, which replaces the inode. The mount keeps
pointing at the old one, so the container silently keeps running the previous code. There is no
error and no restart — the container just serves stale code.

This can leave the file *half* updated. Changing `docker/patches/mcp_server.py` produced a
container where line 691 already read the new value but line 734 still read the old one, with a
checksum that matched neither the old nor the new file.

So after editing a mounted file, verify rather than assume:

```bash
md5sum docker/patches/<file>.py
docker exec docker-ragflow-cpu-1 md5sum /ragflow/<mounted path>
```

Matching checksums confirm the container will load the new code on restart.

## What this fork changes

**Retrieval rules are externalised.** Domain routing (`_domain_rules`), negative-lookahead
patterns (`_NEG_RE`) and the LLM prompt examples were hardcoded in `rag/nlp/search.py`. They now
live in `conf/builtin_retrieval_rules.json` and are loaded through `_brl_*` helpers. See
`docker/patches/BUILTIN_RULES_README.md`. Design points worth knowing:

- `RAGFLOW_BUILTIN_RULES=0` disables the whole mechanism, so a deployment that wants the untouched
  upstream baseline gets it without a code change.
- The config is polled every 30s, so edits apply without a restart.
- `_apply_tenant_ids` scopes the rules to specific tenants; `null` means all.
- Emptying one section (`patterns: []`, `domain_rules: []`) disables only that section.
- The patches in `docker/patches/patch_*.py` are idempotent (each guards with a MARK) and can be
  re-run via `bash docker/patches/apply_builtin_rules.sh`.

**Document management is restricted to the knowledge base's creator.** Upstream lets anyone sharing
the tenant edit a knowledge base, which puts a tenant owner or admin within reach of every knowledge
base in the tenant. `check_kb_manage_permission` in `api/common/check_team_permission.py` narrows
this to the creator alone, and the chunk, document and file-commit APIs route through it. This
removes access — anyone relying on the old team-wide behaviour will lose it.

**The UI is localised.** Chunk-method labels and their descriptions are translated in
`web/src/locales/{zh,en}.ts` instead of showing raw internal keys.

**Smaller changes:** vLLM receives `enable_thinking` under `chat_template_kwargs`; the PDF parser
normalises display labels such as `"Plain Text"` onto the internal keys it dispatches on; MCP runs
in `host` mode so the RAGFlow layer checks each caller's own API key rather than a shared host key.

## Deploying

```bash
cd docker
cp .env .env.local 2>/dev/null || true      # adjust DOC_ENGINE, passwords, RAGFLOW_BUILTIN_RULES
docker compose -f docker-compose.yml up -d
```

Keep `RAGFLOW_BUILTIN_RULES=1` to activate the tuned rules, or `0` for the upstream baseline.

## Upstream

Based on [infiniflow/ragflow](https://github.com/infiniflow/ragflow) v0.26.0, Apache-2.0. The
project description, architecture notes and everything below this notice are upstream's and remain
accurate for this fork except where noted above.

---

<div align="center">
<a href="https://cloud.ragflow.io/">
<img src="web/src/assets/logo-with-text.svg" width="520" alt="ragflow logo">
</a>
</div>

<p align="center">
  <a href="./README.md"><img alt="README in English" src="https://img.shields.io/badge/English-DBEDFA"></a>
  <a href="./README_zh.md"><img alt="简体中文版自述文件" src="https://img.shields.io/badge/简体中文-DFE0E5"></a>
  <a href="./README_tzh.md"><img alt="繁體版中文自述文件" src="https://img.shields.io/badge/繁體中文-DFE0E5"></a>
  <a href="./README_ja.md"><img alt="日本語のREADME" src="https://img.shields.io/badge/日本語-DFE0E5"></a>
  <a href="./README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-DFE0E5"></a>
  <a href="./README_fr.md"><img alt="README en Français" src="https://img.shields.io/badge/Français-DFE0E5"></a>
  <a href="./README_id.md"><img alt="Bahasa Indonesia" src="https://img.shields.io/badge/Bahasa Indonesia-DFE0E5"></a>
  <a href="./README_pt_br.md"><img alt="Português(Brasil)" src="https://img.shields.io/badge/Português(Brasil)-DFE0E5"></a>
  <a href="./README_ar.md"><img alt="README in Arabic" src="https://img.shields.io/badge/Arabic-DFE0E5"></a>
  <a href="./README_tr.md"><img alt="Türkçe README" src="https://img.shields.io/badge/Türkçe-DFE0E5"></a>
</p>

<p align="center">
    <a href="https://x.com/intent/follow?screen_name=infiniflowai" target="_blank">
        <img src="https://img.shields.io/twitter/follow/infiniflow?logo=X&color=%20%23f5f5f5" alt="follow on X(Twitter)">
    </a>
    <a href="https://cloud.ragflow.io" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/Get-Started-4e6b99">
    </a>
    <a href="https://hub.docker.com/r/infiniflow/ragflow" target="_blank">
        <img src="https://img.shields.io/docker/pulls/infiniflow/ragflow?label=Docker%20Pulls&color=0db7ed&logo=docker&logoColor=white&style=flat-square" alt="docker pull infiniflow/ragflow:v0.26.0">
    </a>
    <a href="https://github.com/infiniflow/ragflow/releases/latest">
        <img src="https://img.shields.io/github/v/release/infiniflow/ragflow?color=blue&label=Latest%20Release" alt="Latest Release">
    </a>
    <a href="https://github.com/infiniflow/ragflow/blob/main/LICENSE">
        <img height="21" src="https://img.shields.io/badge/License-Apache--2.0-ffffff?labelColor=d4eaf7&color=2e6cc4" alt="license">
    </a>
    <a href="https://deepwiki.com/infiniflow/ragflow">
        <img alt="Ask DeepWiki" src="https://deepwiki.com/badge.svg">
    </a>
</p>

<h4 align="center">
  <a href="https://cloud.ragflow.io">Cloud</a> |
  <a href="https://ragflow.io/docs/dev/">Document</a> |
  <a href="https://github.com/infiniflow/ragflow/issues/12241">Roadmap</a> |
  <a href="https://discord.gg/NjYzJD3GM3">Discord</a>
</h4>

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://raw.githubusercontent.com/infiniflow/ragflow-docs/refs/heads/image/image/ragflow-octoverse.png" width="1200"/>
</div>

<div align="center">
<a href="https://trendshift.io/repositories/9064" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9064" alt="infiniflow%2Fragflow | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<details open>
<summary><b>📕 Table of Contents</b></summary>

- 💡 [What is RAGFlow?](#-what-is-ragflow)
- 🎮 [Get Started](#-get-started)
- 📌 [Latest Updates](#-latest-updates)
- 🌟 [Key Features](#-key-features)
- 🔎 [System Architecture](#-system-architecture)
- 🎬 [Self-Hosting](#-self-hosting)
- 🔧 [Configurations](#-configurations)
- 🔧 [Build a Docker image](#-build-a-docker-image)
- 🔨 [Launch service from source for development](#-launch-service-from-source-for-development)
- 📚 [Documentation](#-documentation)
- 📜 [Roadmap](#-roadmap)
- 🏄 [Community](#-community)
- 🙌 [Contributing](#-contributing)

</details>

## 💡 What is RAGFlow?

[RAGFlow](https://ragflow.io/) is a leading open-source Retrieval-Augmented Generation ([RAG](https://ragflow.io/basics/what-is-rag)) engine that fuses cutting-edge RAG with Agent capabilities to create a superior context layer for LLMs. It offers a streamlined RAG workflow adaptable to enterprises of any scale. Powered by a converged [context engine](https://ragflow.io/basics/what-is-agent-context-engine) and pre-built agent templates, RAGFlow enables developers to transform complex data into high-fidelity, production-ready AI systems with exceptional efficiency and precision.

## 🎮 Get Started

Try our cloud service at [https://cloud.ragflow.io](https://cloud.ragflow.io).

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://raw.githubusercontent.com/infiniflow/ragflow-docs/refs/heads/image/image/chunking.gif" width="1200"/>
<img src="https://raw.githubusercontent.com/infiniflow/ragflow-docs/refs/heads/image/image/agentic-dark.gif" width="1200"/>
</div>

## 🔥 Latest Updates

- 2026-06-15 Support multiple chat channels such as Feishu, Discord, Telegram, Line, etc.
- 2026-04-24 Supports DeepSeek v4.
- 2026-03-24 [RAGFlow Skill on OpenClaw](https://clawhub.ai/yingfeng/ragflow-skill) — Provides an official skill for accessing RAGFlow datasets via OpenClaw.
- 2025-12-26 Supports 'Memory' for AI agent.
- 2025-11-19 Supports Gemini 3 Pro.
- 2025-11-12 Supports data synchronization from Confluence, S3, Notion, Discord, Google Drive.
- 2025-10-23 Supports MinerU & Docling as document parsing methods.
- 2025-10-15 Supports orchestrable ingestion pipeline.
- 2025-08-08 Supports OpenAI's latest GPT-5 series models.
- 2025-08-01 Supports agentic workflow and MCP.
- 2025-05-23 Adds a Python/JavaScript code executor component to Agent.
- 2025-03-19 Supports using a multi-modal model to make sense of images within PDF or DOCX files.

## 🎉 Stay Tuned

⭐️ Star our repository to stay up-to-date with exciting new features and improvements! Get instant notifications for new
releases! 🌟

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://github.com/user-attachments/assets/18c9707e-b8aa-4caf-a154-037089c105ba" width="1200"/>
</div>

## 🌟 Key Features

### 🍭 **"Quality in, quality out"**

- [Deep document understanding](./deepdoc/README.md)-based knowledge extraction from unstructured data with complicated
  formats.
- Finds "needle in a data haystack" of literally unlimited tokens.

### 🍱 **Template-based chunking**

- Intelligent and explainable.
- Plenty of template options to choose from.

### 🌱 **Grounded citations with reduced hallucinations**

- Visualization of text chunking to allow human intervention.
- Quick view of the key references and traceable citations to support grounded answers.

### 🍔 **Compatibility with heterogeneous data sources**

- Supports Word, slides, excel, txt, images, scanned copies, structured data, web pages, and more.

### 🛀 **Automated and effortless RAG workflow**

- Streamlined RAG orchestration catered to both personal and large businesses.
- Configurable LLMs as well as embedding models.
- Multiple recall paired with fused re-ranking.
- Intuitive APIs for seamless integration with business.

## 🔎 System Architecture

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://github.com/user-attachments/assets/31b0dd6f-ca4f-445a-9457-70cb44a381b2" width="1000"/>
</div>

## 🎬 Self-Hosting

### 📝 Prerequisites

- CPU >= 4 cores
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1
- Python >= 3.13
- [gVisor](https://gvisor.dev/docs/user_guide/install/): Required only if you intend to use the code executor (sandbox) feature of RAGFlow.

> [!TIP]
> If you have not installed Docker on your local machine (Windows, Mac, or Linux), see [Install Docker Engine](https://docs.docker.com/engine/install/).

### 🚀 Start up the server

1. Ensure `vm.max_map_count` >= 262144:

   > To check the value of `vm.max_map_count`:
   >
   > ```bash
   > $ sysctl vm.max_map_count
   > ```
   >
   > Reset `vm.max_map_count` to a value at least 262144 if it is not.
   >
   > ```bash
   > # In this case, we set it to 262144:
   > $ sudo sysctl -w vm.max_map_count=262144
   > ```
   >
   > This change will be reset after a system reboot. To ensure your change remains permanent, add or update the
   > `vm.max_map_count` value in **/etc/sysctl.conf** accordingly:
   >
   > ```bash
   > vm.max_map_count=262144
   > ```
   >
2. Clone the repo:

   ```bash
   $ git clone https://github.com/infiniflow/ragflow.git
   ```
3. Start up the server using the pre-built Docker images:

> [!CAUTION]
> All Docker images are built for x86 platforms. We don't currently offer Docker images for ARM64.
> If you are on an ARM64 platform, follow [this guide](https://ragflow.io/docs/dev/build_docker_image) to build a Docker image compatible with your system.

> The command below downloads the `v0.26.0` edition of the RAGFlow Docker image. See the following table for descriptions of different RAGFlow editions. To download a RAGFlow edition different from `v0.26.0`, update the `RAGFLOW_IMAGE` variable accordingly in **docker/.env** before using `docker compose` to start the server.

```bash
   $ cd ragflow/docker

   # git checkout v0.26.0
   # Optional: use a stable tag (see releases: https://github.com/infiniflow/ragflow/releases)
   # This step ensures the **entrypoint.sh** file in the code matches the Docker image version.

   # Use CPU for DeepDoc tasks:
   $ docker compose -f docker-compose.yml up -d

   # To use GPU to accelerate DeepDoc tasks:
   # sed -i '1i DEVICE=gpu' .env
   # docker compose -f docker-compose.yml up -d
```

> Note: Prior to `v0.22.0`, we provided both images with embedding models and slim images without embedding models. Details as follows:

| RAGFlow image tag | Image size (GB) | Has embedding models? | Stable?        |
|-------------------|-----------------|-----------------------|----------------|
| v0.21.1           | &approx;9       | ✔️                    | Stable release |
| v0.21.1-slim      | &approx;2       | ❌                     | Stable release |

> Starting with `v0.22.0`, we ship only the slim edition and no longer append the **-slim** suffix to the image tag.

4. Check the server status after having the server up and running:

   ```bash
   $ docker logs -f docker-ragflow-cpu-1
   ```

   _The following output confirms a successful launch of the system:_

   ```bash

         ____   ___    ______ ______ __
        / __ \ /   |  / ____// ____// /____  _      __
       / /_/ // /| | / / __ / /_   / // __ \| | /| / /
      / _, _// ___ |/ /_/ // __/  / // /_/ /| |/ |/ /
     /_/ |_|/_/  |_|\____//_/    /_/ \____/ |__/|__/

    * Running on all addresses (0.0.0.0)
   ```

   > If you skip this confirmation step and directly log in to RAGFlow, your browser may prompt a `network abnormal`
   > error because, at that moment, your RAGFlow may not be fully initialized.
   >
5. In your web browser, enter the IP address of your server and log in to RAGFlow.

   > With the default settings, you only need to enter `http://IP_OF_YOUR_MACHINE` (**sans** port number) as the default
   > HTTP serving port `80` can be omitted when using the default configurations.
   >
6. In [service_conf.yaml.template](./docker/service_conf.yaml.template), select the desired LLM factory in `user_default_llm` and update
   the `API_KEY` field with the corresponding API key.

   > See [llm_api_key_setup](https://ragflow.io/docs/dev/llm_api_key_setup) for more information.
   >

   _The show is on!_

## 🔧 Configurations

When it comes to system configurations, you will need to manage the following files:

- [.env](./docker/.env): Keeps the fundamental setups for the system, such as `SVR_HTTP_PORT`, `MYSQL_PASSWORD`, and
  `MINIO_PASSWORD`.
- [service_conf.yaml.template](./docker/service_conf.yaml.template): Configures the back-end services. The environment variables in this file will be automatically populated when the Docker container starts. Any environment variables set within the Docker container will be available for use, allowing you to customize service behavior based on the deployment environment.
- [docker-compose.yml](./docker/docker-compose.yml): The system relies on [docker-compose.yml](./docker/docker-compose.yml) to start up.

> The [./docker/README](./docker/README.md) file provides a detailed description of the environment settings and service
> configurations which can be used as `${ENV_VARS}` in the [service_conf.yaml.template](./docker/service_conf.yaml.template) file.

To update the default HTTP serving port (80), go to [docker-compose.yml](./docker/docker-compose.yml) and change `80:80`
to `<YOUR_SERVING_PORT>:80`.

Updates to the above configurations require a reboot of all containers to take effect:

> ```bash
> $ docker compose -f docker-compose.yml up -d
> ```

### Switch doc engine from Elasticsearch to Infinity

RAGFlow uses Elasticsearch by default for storing full text and vectors. To switch to [Infinity](https://github.com/infiniflow/infinity/), follow these steps:

1. Stop all running containers:

   ```bash
   $ docker compose -f docker/docker-compose.yml down -v
   ```

> [!WARNING]
> `-v` will delete the docker container volumes, and the existing data will be cleared.

2. Set `DOC_ENGINE` in **docker/.env** to `infinity`.
3. Start the containers:

   ```bash
   $ docker compose -f docker-compose.yml up -d
   ```

> [!WARNING]
> Switching to Infinity on a Linux/arm64 machine is not yet officially supported.

## 🔧 Build a Docker image

This image is approximately 2 GB in size and relies on external LLM and embedding services.

```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/
docker build --platform linux/amd64 -f Dockerfile -t infiniflow/ragflow:nightly .
```

Or if you are behind a proxy, you can pass proxy arguments:

```bash
docker build --platform linux/amd64 \
  --build-arg http_proxy=http://YOUR_PROXY:PORT \
  --build-arg https_proxy=http://YOUR_PROXY:PORT \
  -f Dockerfile -t infiniflow/ragflow:nightly .
```

## 🔨 Launch service from source for development

1. Install `uv` and `pre-commit`, or skip this step if they are already installed:

   ```bash
   pipx install uv pre-commit
   ```
2. Clone the source code and install Python dependencies:

   ```bash
   git clone https://github.com/infiniflow/ragflow.git
   cd ragflow/
   uv sync --python 3.13 # install RAGFlow dependent python modules
   uv run python3 download_deps.py
   pre-commit install
   ```
3. Launch the dependent services (MinIO, Elasticsearch, Redis, and MySQL) using Docker Compose:

   ```bash
   docker compose -f docker/docker-compose-base.yml up -d
   ```

   Add the following line to `/etc/hosts` to resolve all hosts specified in **docker/.env** to `127.0.0.1`:

   ```
   127.0.0.1       es01 infinity mysql minio redis sandbox-executor-manager
   ```
4. If you cannot access HuggingFace, set the `HF_ENDPOINT` environment variable to use a mirror site:

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
5. If your operating system does not have jemalloc, please install it as follows:

   ```bash
   # Ubuntu
   sudo apt-get install libjemalloc-dev
   # CentOS
   sudo yum install jemalloc
   # OpenSUSE
   sudo zypper install jemalloc
   # macOS
   sudo brew install jemalloc
   ```
6. Launch backend service:

   ```bash
   source .venv/bin/activate
   export PYTHONPATH=$(pwd)
   bash docker/launch_backend_service.sh
   ```
7. Install frontend dependencies:

   ```bash
   cd web
   npm install
   ```
8. Launch frontend service:

   ```bash
   npm run dev
   ```

   _The following output confirms a successful launch of the system:_

   ![](https://github.com/user-attachments/assets/0daf462c-a24d-4496-a66f-92533534e187)
9. Stop RAGFlow front-end and back-end service after development is complete:

   ```bash
   pkill -f "ragflow_server.py|task_executor.py"
   ```

## 📚 Documentation

- [Quickstart](https://ragflow.io/docs/dev/)
- [Configuration](https://ragflow.io/docs/dev/configurations)
- [Release notes](https://ragflow.io/docs/dev/release_notes)
- [User guides](https://ragflow.io/docs/category/user-guides)
- [Developer guides](https://ragflow.io/docs/category/developer-guides)
- [References](https://ragflow.io/docs/dev/category/references)
- [FAQs](https://ragflow.io/docs/dev/faq)

Everything below this line is upstream's, unchanged.

## 📜 Roadmap

See the [RAGFlow Roadmap 2026](https://github.com/infiniflow/ragflow/issues/12241)

## 🏄 Community

- [Discord](https://discord.gg/NjYzJD3GM3)
- [X](https://x.com/infiniflowai)
- [GitHub Discussions](https://github.com/orgs/infiniflow/discussions)

## 🙌 Contributing

RAGFlow flourishes via open-source collaboration. In this spirit, we embrace diverse contributions from the community.
If you would like to be a part, review our [Contribution Guidelines](https://ragflow.io/docs/dev/contributing) first.

## Licensing

Upstream RAGFlow is Apache-2.0, and this fork keeps that licence — see [LICENSE](LICENSE). The
retrieval rules in `docker/patches/builtin_retrieval_rules.json` were written for one specific
deployment and encode that organisation's product lines, document naming and part numbers. Nothing
here is a claim of authorship over RAGFlow itself; credit belongs to
[InfiniFlow](https://github.com/infiniflow/ragflow) and its contributors.

If you reuse this, expect to replace `docker/patches/` with your own tuning, and read
[What this fork changes](#what-this-fork-changes) first — the permission model is *narrower* than
upstream's, which will break anything that relies on team-wide knowledge base access.
