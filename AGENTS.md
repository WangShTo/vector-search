# vector-search —— 项目约定

## 是什么

自研 HNSW + BM25 的检索服务：**Java 21 引擎 + Python 模型服务，双进程**，不用 Faiss / Milvus / Elasticsearch。
简历向项目，所以每个数字都要能复现、每处取舍都要能解释。

路径 `G:\all workspace\vector-search`（先后待过 `G:\dsh-workspace\vector-search`、`G:\dsh-workspace`）。
**目录一改名，会话工作目录就失效**（`BLOCKED by CC Safety Net … cwd.session-unusable`），移动/改名前后各说一句。

## 组成

| 路径 | 是什么 |
|---|---|
| `vector-search-java/` | Spring Boot 3 + JDK 21 引擎：HNSW、BM25、RRF 融合、精排编排、快照/压实、REST、JMH 与召回基准 |
| `model-service/` | FastAPI + `bge-small-zh-v1.5`（**512 维**）+ `bge-reranker`：编码、动态批处理、精排打分 |
| `vector-search-java/src/main/resources/static/` | 前端三页，**原生 HTML/CSS/JS 无构建**：`#/search`、`#/library`、`#/ops` |
| `scripts/` | `dev.ps1` `start\|stop\|status`、`run_bench.ps1`、`load_corpus.py`、`render_bench_report.py`、`smoke_test.py` |
| `bench/` | `REPORT.md`（**脚本生成，禁止手改**）+ `results/*.json`（原始数据） |
| `docs/` | `RESUME.md`（简历文案 + 面试问答）、`COMPETITORS.md`（竞品对比）、`DEPLOY.md`、`architecture.*` |
| `data/corpus/` | 评测语料（4 个 md，切块后约 602 块，含代码，用于 recall 基准） |
| `data/corpus-demo/` | 演示语料（7 个 md，切块后约 124 块，纯散文，剥掉代码块和表格）—— **两份语料故意不同** |
| `deploy/`、`docker-compose.yml` | 容器编排；`deploy/docker-compose.prod.yml` 面向公网 |

端口：Java `8080`，Python `8000`。

## 怎么跑

```powershell
cd 'G:\all workspace\vector-search'
.\scripts\dev.ps1 start      # 起两个服务，等就绪
.\scripts\dev.ps1 status
.\scripts\dev.ps1 stop
```

**必须 JDK 21**（Vector API `jdk.incubator.vector` 在 17 上根本不存在，不是"性能差一点"而是编不过）。
`JAVA_HOME=C:\Users\wst21\tools\jdk-21`，Maven 在 `C:\Users\wst21\tools\maven`。

测试与基准：

```powershell
cd vector-search-java; mvn -B -ntp test        # 57 个
cd model-service; .\.venv\Scripts\python.exe -m pytest tests -q   # 10 个
.\scripts\run_bench.ps1                        # 重算 bench/REPORT.md
```

## 改代码前必须知道

1. **维度 512 要三处一致**（`application.yml`、模型输出、已有快照）。不一致时不会报错、只会算错 —— 所以入口处有硬校验，别绕过。
2. **快照有格式版本号**（`HnswSnapshot`，当前 v3）。改结构必须升版本，同时保证旧版本仍可读；否则老用户升级即丢索引。
3. **`data-dir` 是相对进程工作目录解析的**，启动日志会打绝对路径。从 `vector-search-java/` 启动和从仓库根启动落在不同地方。
4. **`bench/REPORT.md` 是脚本产物**。改了 `bench/results/*.json` 的结构，必须同步改 `scripts/render_bench_report.py`，再跑一遍，不要手改。
5. **前端没有构建步骤**，改完刷新页面即可；静态资源在 `static/` 根目录，不在子目录里（`/app/` 是 404）。

## 验收（"完成"的定义）

- `mvn test` 57 个全绿、`pytest` 10 个全绿
- `dev.ps1 start` 后 `/api/stats` 有数据、能检索出结果
- **改检索/索引逻辑必须附基准数字的前后对比**（`bench/REPORT.md` 或临时脚本都行，但不能只有"编译通过"）
- 完成 = 证据 + 一行 `未验证：`

## 这台机器的坑（都真实踩过）

1. **没有 `pwsh`**，只有 Windows PowerShell 5.1。`& pwsh -File ...` 报 `CommandNotFoundException`，直接 `& .\scripts\dev.ps1 stop`。
2. **`.ps1` 必须 UTF-8 BOM**。否则 5.1 按 GBK 解码，中文把引号吃掉，报语法错。
3. **管道接 `Select-String` 会假报 `[exit code: 1]`**。真实结果看 `$LASTEXITCODE`，或把输出重定向到日志再解析。
4. **禁止用 `Get-Content`/`Set-Content` 往返源码文件** —— 5.1 按 GBK 读、加 BOM 写，会把中文全变成 `U+FFFD`（曾经毁掉整个 `HnswIndex.java`，只能从 git 恢复）。改文件只用 `edit`/`write` 工具，或 `[System.IO.File]` 显式指定 UTF8。
5. **并发基准在这台机器上不可信**：i7-14650HX 是 8 P + 8 E 混合核，同一 8 线程配置跨轮次测出 20k / 28k / 35k / 42k / 82k ops/s（4 倍差）。**不要在简历或 README 里写并发数字**，报告第四节已经把原因写明。
6. **网络**：HuggingFace 要 `HF_ENDPOINT=https://hf-mirror.com`；pip 用 `mirrors.aliyun.com/pypi/simple`；Maven 用 aliyun 镜像；`raw.githubusercontent.com` **在本机超时**，读 GitHub README 用 `api.github.com/repos/{o}/{r}/readme` + `Accept: application/vnd.github.raw`。
7. **`git push` 的输出走 stderr**，PowerShell 会当成 `NativeCommandError`，忽略即可，看 `old..new main -> main` 那行才是真结果。

## 方法学（这个项目的立身之本）

- **先测再优化，测不出来就不改**。例：有人（包括未来的你）会想换 `StampedLock`，实测无竞争读锁 16 ns，占检索延迟 0.015% —— 不值得，明确否决并留档。
- **两套延迟不能混**：JMH 是稳态真值；RecallRunner 为了配对召回会系统性偏高（104.5 µs → 119.7 µs）。引用时标明出处。
- **报数字必须带边界**。量化省了 3.35 倍内存但延迟涨 40%，就说涨 —— 主动交代比被追问出来强。
- **反直觉的结果优先保留**：混合检索在词面查询上不如纯 BM25 —— 这条比"混合最好"值钱，它证明的是会做实验而不是会调库。

## 禁区

- 不改目录结构、不改项目名；移动前先说。
- 不手改生成物（`bench/REPORT.md`）。
- 密钥只走环境变量（`VECTOR_SEARCH_API_KEY`），不写进 yml、日志或提交。
- 不提交 `data/index/`（快照）、`.venv`、模型权重。
- `git push` 已授权（他要求每次更新同步 GitHub）；但 **force push、删分支、改仓库设置** 必须先问。
- 不覆盖他没让动的模块；不动 `data/corpus*` 的切块结果（一改所有召回数字作废）。
