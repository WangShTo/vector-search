# 固定 3.11：torch 在这个版本上的 wheel 最齐全，
# 用 3.13 有时会遇到某些依赖还没有预编译包、要现场编译的情况
FROM python:3.11-slim

WORKDIR /app

# 先装依赖再拷代码：只要 requirements.txt 没变，改代码就不会触发重装，
# 镜像重建从几分钟降到几秒
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 模型在首次启动时从 HuggingFace 下载，缓存到 /root/.cache/huggingface，
# docker-compose 里挂成 named volume，重建容器不会重新下载
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 单 worker 是刻意的：动态批处理靠进程内的共享队列攒批，
# 多 worker 会让每个进程各攒一批，batch 永远攒不大（见 /stats 的平均批大小）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
