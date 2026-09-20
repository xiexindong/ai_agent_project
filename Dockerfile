# ====================================================================
# 应用镜像：把「能跑」变成「谁都能跑起来」
#   面试官/同事 clone 下来只需要 docker compose up，不用猜 Python 版本和依赖。
#
# 构建：docker build -t enterprise-agent:latest .
# 运行：docker run --env-file .env -p 8000:8000 enterprise-agent:latest
# ====================================================================
FROM python:3.11-slim

# 【为什么固定 3.11 而不是用 latest】
#   pymilvus / psycopg2 等 C 扩展在各版本上的 wheel 覆盖不一致，
#   latest 一旦跳版本就会出现"本地能跑、CI 装不上"。钉死版本 = 可复现。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：gcc 用于少数没有 wheel 的包编译兜底；libpq 是 psycopg2 的运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 【为什么先只拷 requirements.txt】
#   Docker 按层缓存：依赖装好后，改业务代码不会触发重装依赖，
#   二次构建从 3 分钟降到 10 秒。把 COPY . . 放前面就享受不到这个缓存。
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# 非 root 运行：容器被攻破时不至于拿到宿主机 root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# 健康检查用 /health（不依赖数据库，容器编排靠它判断存活）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "main.py", "serve"]
