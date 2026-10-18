# 固定 3.11：torch 在这个版本上的 wheel 最齐全，
# 用 3.13 有时会遇到某些依赖还没有预编译包、要现场编译的情况
FROM python:3.11-slim

WORKDIR /app

# torch 单独装，而且走 CPU 专用索引。
# 直接从 PyPI 装 torch 会拖下来 2 GB+ 的 CUDA 运行库 —— 这个镜像只跑 CPU，
# 那些库一个都用不上，镜像体积会从约 2 GB 涨到 7 GB 以上。
# 这一步放在 requirements 之前，是为了让它自己占一层、能被缓存。
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# 先装依赖再拷代码：只要 requirements.txt 没变，改代码就不会触发重装，
# 镜像重建从几分钟降到几秒。
# requirements.txt 里的 torch 是下限约束（>=2.4），上面已经装好 CPU 版，
# pip 会认为已满足而跳过 —— 不会把它偷偷换成 CUDA 版。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 模型在首次启动时从 HuggingFace 下载，缓存到 /root/.cache/huggingface，
# docker-compose 里挂成 named volume，重建容器不会重新下载
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 单 worker 是刻意的：动态批处理靠进程内的共享队列攒批，
# 多 worker 会让每个进程各攒一批，batch 永远攒不大（见 /stats 的平均批大小）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
