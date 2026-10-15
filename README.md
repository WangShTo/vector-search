# 自研向量检索服务（Java 引擎 + Python 模型服务）

从零实现的语义检索服务：**Java 侧自研 HNSW 索引引擎**负责存储与检索，**Python 侧**提供 embedding / rerank 模型服务。不依赖 Faiss、Milvus、Elasticsearch 等任何现成向量检索库。

> 目标不是"跑通一个 RAG demo"，而是**把检索系统的每一步都做成可测量、可对比、可解释的工程决策**。

---

## 目录

- [这个项目在做什么](#这个项目在做什么)
- [架构](#架构)
- [快速开始](#快速开始)
- [API 示例](#api-示例)
- [设计取舍（面试弹药）](#设计取舍面试弹药)
- [评测方法](#评测方法)
- [路线图](#路线图)
- [目录结构](#目录结构)

---

## 这个项目在做什么

语义检索要解决的核心矛盾是：**在百万级文档里，用毫秒级延迟，找到语义最相关的几条。**

暴力遍历所有向量做内积，准确率 100%，但 O(n·d) 的复杂度在百万级数据上完全不可用。HNSW（Hierarchical Navigable Small World）用近似换速度：把向量组织成分层邻近图，用少量召回率损失换取数量级的延迟下降。

这个项目把这条链路完整实现了一遍，并且**每一步优化都保留可对比的基线**：

| 环节 | 基线 | 优化后 | 验证方式 |
|---|---|---|---|
| 检索结构 | 暴力遍历 | HNSW 分层图 | recall@10 对照暴力检索 |
| 距离计算 | 标量循环 | JDK Vector API (SIMD) | JMH 微基准 |
| 向量存储 | `List<float[]>` 堆内 | 堆外内存（计划中） | 堆占用与 GC 停顿 |
| 跨进程调用 | 逐条 embedding | 批量 + 动态批处理 | 端到端 P95 延迟 |
| 并发读 | `ReentrantReadWriteLock` | `StampedLock` 乐观读（计划中） | 压测 QPS / P99 |

**关键点：没有基线就没有优化。** 所有"提升 N 倍"的结论，都能在这个仓库里找到对照组。

---

## 架构

```
                          ┌────────────────────────────────────────────────┐
   客户端 ───────────────►│  Java 检索服务   (Spring Boot 3 / Java 21)      │
   HTTP / JSON            │                                                │
                          │  ┌──────────────────────────────────────────┐  │
                          │  │ HNSW 索引引擎（自研）                     │  │
                          │  │  · 分层邻近图 / 贪心下降 / 启发式邻居选择  │  │
                          │  │  · Vector API SIMD 距离计算               │  │
                          │  └──────────────────────────────────────────┘  │
                          │  ┌──────────────────────────────────────────┐  │
                          │  │ 暴力检索索引（正确性与性能的对照组）       │  │
                          │  └──────────────────────────────────────────┘  │
                          │  ┌──────────────────────────────────────────┐  │
                          │  │ BM25 倒排 + RRF 混合召回（第 4 周）        │  │
                          │  └──────────────────────────────────────────┘  │
                          │  文档原文存储 / 分阶段耗时统计                  │
                          └───────────────────┬────────────────────────────┘
                                              │  HTTP（批量调用）
                                              ▼
                          ┌────────────────────────────────────────────────┐
   评测脚本 ─────────────►│  Python 模型服务   (FastAPI)                    │
   eval/evaluate.py       │                                                │
                          │  · BGE-small-zh 句向量（512 维）                │
                          │  · 动态批处理（攒批，吞吐换延迟）                │
                          │  · Cross-Encoder 精排（懒加载）                 │
                          └────────────────────────────────────────────────┘
```

**为什么索引在 Java、模型在 Python？** 这是刻意的职责切分。模型推理生态在 Python 无可替代；而索引引擎需要的是内存布局控制、并发原语、JIT 优化——那正是 Java 的强项。两者的边界（跨进程批量调用）本身就是本项目最重要的性能优化对象。

---

## 快速开始

### 前置要求

| 组件 | 版本 | 说明 |
|---|---|---|
| JDK | **21**（17 亦可） | Spring Boot 3 需要 17+；Vector API 需要 16+ |
| Maven | 3.9+ | |
| Python | 3.11 – 3.13 | 3.13 已验证可用 |
| Docker | 可选 | 用于一键部署 |

### 方式一：本地分别启动（推荐用于开发）

```bash
# ---------- 1. Python 模型服务 ----------
cd model-service
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS

.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# 首次启动会下载 BGE 模型（约 400MB），之后走本地缓存
```

```bash
# ---------- 2. Java 检索服务 ----------
cd vector-search-java
mvn spring-boot:run
```

服务地址：Java `http://localhost:8080`，Python `http://localhost:8000`。

### 方式二：Docker Compose

```bash
docker compose up --build
```

### ⚠️ 两个国内环境的坑

这两个问题都会表现为**静默挂起**，而不是明确的报错，所以单独写出来：

**1. HuggingFace 直连超时。** `huggingface.co` 在国内无法访问，模型下载会一直挂着直到超时。项目已在 `model-service/app/config.py` 里默认设置镜像：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

这段代码**必须在导入 `huggingface_hub` 之前执行**，所以 `config` 模块要最先被加载。调整 `main.py` 的导入顺序时务必保持这一点。

**2. Adoptium 官网下载极慢。** `api.adoptium.net` 会跳转到 GitHub Releases，实测只有 ~32 KB/s。建议改用 Microsoft Build of OpenJDK（同一 CDN 实测 16 MB/s）：

```powershell
Invoke-WebRequest 'https://aka.ms/download-jdk/microsoft-jdk-21-windows-x64.zip' -OutFile jdk21.zip
```

---

## API 示例

### 批量灌入文档

```bash
curl -X POST http://localhost:8080/api/documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": 1, "text": "HNSW 是一种基于分层可导航小世界图的近似最近邻算法。"},
      {"id": 2, "text": "BM25 是经典的稀疏检索算法，基于词频和逆文档频率。"}
    ]
  }'
```

### 检索

```bash
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "分层图的近似最近邻", "topK": 5}'
```

响应里带**分阶段耗时**，这是定位瓶颈的关键：

```json
{
  "query": "分层图的近似最近邻",
  "hits": [{"id": 1, "score": 0.8731, "text": "HNSW 是一种..."}],
  "indexType": "hnsw",
  "timing": {"embeddingMs": 45, "searchMs": 2, "totalMs": 47}
}
```

> 如果 `embeddingMs` 远大于 `searchMs`，说明瓶颈在跨进程调用而不是索引本身——
> 这时候去调 HNSW 参数是南辕北辙。这个字段就是为了让这种误判不可能发生。

### 查看索引状态

```bash
curl http://localhost:8080/api/stats
```

返回 HNSW 的图结构指标（层高直方图、总边数）。**层高分布应该呈金字塔状衰减**，如果不是，说明建图参数或随机源有问题，recall 一定上不去。

### 切换索引实现（做对照实验用）

```bash
curl -X POST "http://localhost:8080/api/admin/reset?type=brute-force"
```

> ⚠️ 会清空当前索引。这是刻意的：对照实验必须在同一份数据、干净的状态下重建，否则测出来的数字不可比。

---

## 设计取舍（面试弹药）

这一节记录**每个决策背后的理由和被否决的方案**。代码写对不难，能讲清"为什么这么选"才是区分度。

### 1. 为什么邻居选择要用启发式，而不是取最近的 M 个？

直接取最近的 M 个邻居，会让所有邻居聚在同一个方向，查询时容易一头扎进局部最优出不来。HNSW 论文的启发式加了多样性约束：只有当一个候选**距离查询点比距离任一已选邻居都更近**时才收它——直觉上是"这个邻居提供了别人提供不了的方向"。

不足 M 个时用被丢弃候选中最近的补齐（`keepPrunedConnections`），保证连通度。

### 2. 为什么 layer 0 的最大出边数是上层的两倍？

底层承载全部节点，需要更高连通度才能保证搜索不走进死胡同；上层节点数指数级减少，用 M 就够，也更省内存。这是内存与连通度的权衡。

### 3. 为什么 `efConstruction` 可以放心调大，`m` 不行？

`efConstruction` 只影响**建图时**的候选池大小，即一次性的构建成本，对查询延迟零影响。
`m` 决定了图的边数上限，一旦索引建好，改 `m` 就意味着必须重建整个索引。

**所以线上调优的策略是：`m` 一次性定好（16~48），把 `efSearch` 作为运行时可调的旋钮。**

### 4. 为什么查询向量要归一化，且用内积而不是余弦距离？

归一化后内积等价于余弦相似度，但省掉了每次比较时的两次开方和一次除法。在千万次比较的热点路径上，这个常数因子不可忽略。Python 侧 `normalize_embeddings=True` 和 Java 侧 `normalizedCopy` 是幂等的双重保险。

### 5. 为什么 embedding 接口设计成批量的？

Java 和 Python 是两个进程，每次调用都是一次完整的网络往返。灌 10 万条文档时，逐条调用就是 10 万次往返——HTTP 和 TCP 的开销会远超模型推理本身。批量接口把 N 次往返压缩成 1 次。

这也是为什么检索接口虽然只处理一条 query，却仍然走批量代码路径：**保持单一代码路径，避免"批量对了单条错了"这类只在某条路径上出现的 bug。**

### 6. 动态批处理为什么是单 worker？

动态批处理靠**进程内的共享队列**攒批。uvicorn 开多 worker 时每个进程各有一个队列，请求被打散，batch 永远攒不大——`/stats` 里的"平均批大小"长期接近 1 就是这个症状。

代价是单 worker 无法利用多核。当前的取舍是：**批处理带来的吞吐收益 > 多进程并行带来的收益**，因为模型推理是主要的 CPU 消耗方，而它已经吃满了。

### 7. 为什么 `SearchResult` 对外用相似度而不是距离？

HNSW 内部按"距离越小越近"排序，而归一化后的内积"越大越相似"。内部统一转换成距离语义做比较，对外统一用相似度。**两套方向混用是这类系统最容易埋下的 bug**，所以在类型层面就固定下来。

---

## 评测方法

评测集自己构建，因为通用中文数据集和你的语料领域不匹配，而"recall@10 达到 0.9"这句话只有在**你自己的语料和标注**上才有意义。

```bash
# 1. 把语料放进 data/corpus/（课程讲义、论文、笔记、代码都行）
# 2. 构建评测集
cd eval
python build_dataset.py --corpus-dir ../data/corpus --out ../data/eval --strategy paragraph

# 3. 灌入语料后跑评测
python evaluate.py --queries ../data/eval/queries.jsonl --top-k 10 --label hnsw-m16
```

### 评测协议：已知项检索

对语料里的一个分块，从它自己身上取一段文字当查询，正确答案就是它所在的文档。这样**不需要人工标注相关性**，但要求检索系统真的能把它找回来。

**必须说明的局限**：查询词与文档词高度重合，所以指标是**上界**，不等于线上效果。它适合做**相对比较**（改参数前后、两种索引之间、两种分块策略之间），不适合宣称绝对质量。报告模板里已经写明了这一点——不写清楚就是在误导人。

### 指标

| 指标 | 衡量什么 | 为什么不能只看它 |
|---|---|---|
| recall@k | 相关文档有没有进前 k | "全召回但排序很差"也能拿高分 |
| MRR@k | 第一个相关结果排在第几位 | 只看第一个，忽略其余 |
| NDCG@k | 位置加权的排序质量 | — |
| P50/P95/P99 延迟 | 长尾体验 | **平均值会掩盖长尾，线上体验由 P99 决定** |

---

## 路线图

| 周 | 目标 | 产出 |
|---|---|---|
| 1 | 端到端跑通（正确但慢） | 暴力检索 baseline + HNSW + 服务化 |
| 2 | **量化 HNSW 相对暴力检索的收益** | JMH 基准 + recall/延迟 对照表 |
| 3 | 内存优化 | 堆外内存 + 扁平数组布局 + visited 版本号戳 |
| 4 | 混合召回 | BM25 倒排 + RRF 融合 + rerank 两阶段 |
| 5 | SIMD 与调参 | Vector API + `m`/`efConstruction`/`efSearch` 消融实验 |
| 6 | 并发与持久化 | StampedLock 乐观读 + 索引落盘 + 热更新 + 压测 |
| 7 | 工程化 | Docker 一键起 + CI + 测试覆盖 + README/demo |
| 8 | 输出 | benchmark 报告 + 技术博客 + 简历文案 |

### 当前进度

- [x] HNSW 索引（分层图 / 贪心下降 / 启发式邻居选择）
- [x] 暴力检索 baseline
- [x] 标量距离计算 + Vector API SIMD 实现
- [x] Spring Boot 服务化 + 分阶段耗时统计
- [x] Python 模型服务 + 动态批处理
- [x] 评测脚手架（语料构建 / 查询生成 / 指标计算）
- [x] 单元测试（recall 对照暴力检索）
- [ ] JMH 基准
- [ ] 堆外内存改造
- [ ] BM25 + RRF 混合召回
- [ ] 索引持久化与热更新
- [ ] Docker 镜像验证 + CI

---

## 目录结构

```
vector-search/
├── vector-search-java/              # Java 检索服务
│   └── src/
│       ├── main/java/com/vectorsearch/
│       │   ├── core/                # 索引核心（与框架无关）
│       │   │   ├── DistanceMetrics.java        # 标量距离（基线）
│       │   │   ├── DistanceMetricsSimd.java    # SIMD 距离
│       │   │   ├── VectorIndex.java            # 索引抽象
│       │   │   ├── bruteforce/                 # 暴力检索（对照组）
│       │   │   └── hnsw/                       # HNSW 实现
│       │   ├── client/              # 调 Python 模型服务
│       │   ├── service/             # 索引管理 / 文档存储 / 检索编排
│       │   ├── api/                 # REST 接口与 DTO
│       │   └── config/              # 配置绑定
│       └── test/java/               # 单元测试
├── model-service/                   # Python 模型服务
│   └── app/
│       ├── config.py                # 配置 + HF 镜像设置（必须在最前）
│       ├── embedder.py              # 动态批处理 + BGE embedding
│       ├── reranker.py              # Cross-Encoder 精排（懒加载）
│       ├── schemas.py               # HTTP 契约
│       └── main.py                  # FastAPI 应用
├── eval/                            # 评测脚手架
│   ├── build_dataset.py             # 语料 -> 分块 -> 查询
│   └── evaluate.py                  # recall / MRR / NDCG / 延迟
├── bench/                           # JMH 基准与压测报告（第 2 周起）
├── data/                            # 语料与索引数据（不入库）
└── docker-compose.yml
```
