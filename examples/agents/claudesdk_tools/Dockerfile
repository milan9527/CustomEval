FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_SYSTEM_PYTHON=1 UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 DOCKER_CONTAINER=1 AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
# ripgrep + certs for the bundled Claude Code CLI subprocess
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates ripgrep && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN uv pip install -r requirements.txt
RUN uv pip install aws-opentelemetry-distro==0.12.2
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore
ENV HOME=/home/bedrock_agentcore
EXPOSE 8080
COPY . .
CMD ["opentelemetry-instrument", "python", "-m", "agent"]
