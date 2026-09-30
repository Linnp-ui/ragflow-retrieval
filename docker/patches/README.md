# ragflow 外挂补丁快照

这个目录是 **`docker/patches/` 里的 Python 文件**（外加 `_mounted/` 下两个符号链接），
它们通过 `docker-compose.yml` 的 bind 挂载覆盖容器内路径，是检索、解析、表格、
MCP 等业务逻辑的**真实载体**。

镜像只提供 Python 依赖环境，不含这些逻辑：

- 镜像 tag 里的 `patched` / `agg` / `agg-v2` 字样**没有实际含义**——历史上有过
  三个 ragflow 镜像，功能完全等价；真正生效的是本目录。
- 已删除 `v0.26.0` 与 `v0.26.0-patched-agg-20260826` 两个误导 tag，
  只保留运行中的 `v0.26.0-patched`。
- 重建镜像**不能**找回这里的任何代码；这里丢了就是永久丢失。

## 危险点

容器只 bind 挂载了文件列表里那几项。**如果本目录被删除或某个文件缺失，
容器仍然 Up、检索/解析静默退回上游行为，不会有任何报错。**

因此本目录单独 `git init`（而不是提交进上游 ragflow 仓库）：

- 上游 13 个月历史不被污染，`git pull` 不冲突
- 父仓库的 `git clean -fd` **会跳过嵌套仓库**（git 不删 nested repo）

`_mounted/` 用符号链接指向 `../entrypoint.sh` 和 `../service_conf.yaml.template`，
这样版本控制里的内容与实际挂载的文件永远一致（无副本漂移）。
