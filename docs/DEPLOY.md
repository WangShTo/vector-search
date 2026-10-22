# 部署指南

把两个服务跑在一台公网服务器上，得到一个可以发给别人的地址。

> **先说清楚**：本文档里的 `deploy/docker-compose.prod.yml` **没有在真实 Docker 环境跑过** ——
> 开发机上没有安装 Docker。它经过了静态审查（路径、端口、依赖顺序、健康检查），
> 但「文件正确」和「跑起来没问题」是两件事。按下面的步骤走，遇到问题把报错贴出来。

---

## 一、先算内存账（决定买什么配置）

两个服务都是内存大户，配置买小了会 OOM 而不是变慢。

| 组件 | 常驻内存 | 说明 |
|---|---|---|
| Python 进程 + torch（CPU） | ~400 MB | torch 运行时本身 |
| BGE embedding 模型 | ~120 MB | bge-small-zh-v1.5 |
| **BGE reranker** | **~1.5 GB** | **懒加载**：不开启精排就永远不会加载 |
| JVM（Spring Boot） | ~400 MB | 堆 + 元空间 |
| 向量索引（124 条演示数据） | 可忽略 | 10 万条时约 280 MB |
| 操作系统 + Docker | ~300 MB | |
| **合计（不开精排）** | **~1.3 GB** | |
| **合计（开精排后）** | **~2.8 GB** | |

**结论：买 2 核 4G。** 2G 会紧张（一开精排就 OOM），4G 有余量。

磁盘 **40 GB 起步**：镜像约 3 GB，模型 1.5 GB，索引数据几百 MB。

---

## 二、买服务器

国内几家都行：阿里云 / 腾讯云 / 华为云。挑的时候注意三点：

1. **内存 ≥ 4 GB** —— 这是硬指标，不是建议
2. **系统选 Ubuntu 22.04 或 24.04** —— 下面的命令都按 Ubuntu 写
3. **带宽 3 Mbps 够用** —— 这个项目返回的是文本，不是图片

> **要不要买域名 / 备案？**
> 不要。用 `http://公网IP:8080` 直接访问就行。
> 备案针对的是「域名 + 80/443 端口」，用 IP 加高端口不触发。

---

## 三、部署

### 1. 登录服务器

```bash
ssh root@你的公网IP
```

### 2. 安装 Docker

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
docker --version
```

> 如果 `get.docker.com` 太慢，可以换国内镜像源：
> ```bash
> curl -fsSL https://get.docker.com | sh -s -- --mirror Aliyun
> ```

### 3. 拉代码

```bash
# 用 HTTPS 拉（服务器上没有配 SSH key）
cd /opt
git clone https://github.com/WangShTo/vector-search.git
cd vector-search
```

### 4. 启动

```bash
docker compose -f deploy/docker-compose.prod.yml up -d --build
```

**第一次构建要 10~20 分钟**：下载 Python 依赖（CPU 版 torch 约 200 MB）+ Maven 依赖。

构建完成后看进度：

```bash
docker compose -f deploy/docker-compose.prod.yml logs -f
```

**首次启动还要下载 embedding 模型（约 400 MB）**，走的是 `hf-mirror.com` 镜像。
看到下面这行才算真正就绪：

```
模型服务已就绪，监听 0.0.0.0:8000
```

### 5. 开放端口

两处都要开，少一处都连不上：

1. **云厂商控制台的「安全组」/「防火墙」** → 放行 TCP 8080
2. **服务器本机的 ufw**（如果启用了）：
   ```bash
   ufw allow 8080/tcp
   ```

### 6. 验证

```bash
# 在服务器上
curl -s http://localhost:8080/api/stats | head -c 400

# 在你自己的电脑上
http://公网IP:8080
```

页面能打开、能检索，就成功了。

### 7. 导入你自己的资料

打开页面 → 「知识库」→ 拖入 `.md` / `.txt` 文件。

导入后点「索引运维」→「保存快照」，这样重启容器不用重建索引。

---

## 四、常见问题

### 容器起来就退出，日志显示 `Killed`

内存不够。确认配置 ≥ 4 GB，或者先用 2G 跑、把精排功能关掉（默认就是关的，
除非你在请求里传 `rerank: true`）。

### `curl localhost:8080` 通，但外网打不开

按顺序排查：

```bash
# 1. 服务真的在监听吗
ss -tlnp | grep 8080

# 2. 本机防火墙
ufw status

# 3. 云控制台的安全组（这一步最容易漏）
```

第 3 步是外部控制面，SSH 里看不到，必须去网页控制台。

### 模型下载卡住

```bash
docker compose -f deploy/docker-compose.prod.yml logs model-service | tail -50
```

如果卡在 `Downloading`，检查 compose 里的 `HF_ENDPOINT=https://hf-mirror.com` 是否生效。
实在不行，可以先把模型下载到宿主机再挂载进容器。

### 换机器 / 数据迁移

索引数据在 Docker volume 里，卷名是 `vs-prod-index-data`（在 compose 里显式命名的，
不会随目录名变化）：

```bash
# 备份
docker run --rm -v vs-prod-index-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/index-data.tar.gz -C /data .

# 恢复（新机器上）
docker run --rm -v vs-prod-index-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/index-data.tar.gz -C /data
```

---

## 五、这个部署的安全边界

**这是一个演示部署，不是生产环境。** 上线前请明确知道：

| 现状 | 风险 |
|---|---|
| **没有任何鉴权** | 任何人都能上传、删除文档、清空知识库 |
| 没有 HTTPS | 明文传输（演示内容不敏感，可以接受） |
| 没有限流 | 有人恶意刷检索会打满 CPU |
| 单容器、无备份 | 挂掉要手工重启（`restart: unless-stopped` 会兜底） |

**最低限度的加固**（如果你要让陌生人访问）：

1. 关掉「清空知识库」入口，或者干脆别把「索引运维」页暴露出去
2. 在云控制台的安全组里只放行你自己的 IP
3. 或者用 `nginx` 加一个 Basic Auth，5 分钟能配好

如果你只是想给面试官看，**建议做法**：平时关掉，需要展示时再开机，
把地址和一句说明发过去，看完关掉。

---

## 六、备选：免费的 Hugging Face Spaces

不想花钱的话，HF Spaces 的免费 CPU 额度是 **2 vCPU / 16 GB 内存**，跑这个项目绰绰有余。

代价是要把两个服务塞进**一个镜像**（Spaces 只能起一个容器），并且要让它们监听 Spaces
要求的 7860 端口 —— 大致是：Python 服务监听 8000，Java 监听 8080，
再用 nginx 在 7860 上把请求代理到 8080。

另一个问题是 **`.hf.space` 域名在国内时通时不通**，你自己访问可能都要重试几次。

如果决定走这条路，告诉我，我把合并镜像和 nginx 配置写出来。
