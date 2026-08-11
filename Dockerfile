# Jarvisman backend -- headless CLI + JSON API.
#
# The desktop GUI is NOT built into this image. Qt in a container needs X11
# forwarding, which is fragile across machines and buys nothing: the GUI is a
# thin client (RAG_BACKEND_URL) and runs natively on the workstation. Keeping
# Qt out drops ~150 MB and a dozen X libraries of attack surface from the
# image that actually faces the network.
#
# Build:  docker compose build
# Runs on x86_64 and aarch64 (the DGX Spark) -- no arch-specific steps.

# ---- stage 1: build wheels -------------------------------------------------
# Compilers live here and are thrown away, so the runtime image carries only
# the installed packages.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- stage 2: runtime ------------------------------------------------------
FROM python:3.12-slim AS runtime

# curl for the compose healthcheck; libgomp1 is required by faiss-cpu.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

# Run as a normal user. The sandbox executes model-generated pandas code; if
# anything ever escapes it, it escapes as an unprivileged account that owns
# nothing but /data. UID 1000 matches the usual host user so the bind-mounted
# index stays writable -- override with --build-arg UID=$(id -u) if yours
# differs.
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" jarvis \
    && useradd -m -u "${UID}" -g "${GID}" -s /usr/sbin/nologin jarvis

WORKDIR /app
COPY --chown=jarvis:jarvis . .

# Persisted index; mount a volume to keep it across containers.
ENV RAG_DATA_DIR=/data
RUN mkdir -p /data && chown jarvis:jarvis /data
VOLUME ["/data"]

# Ollama runs as a sibling service (see docker-compose.yml).
ENV OLLAMA_HOST=http://ollama:11434

USER jarvis

ENTRYPOINT ["python", "-m", "jarvisman.cli"]
CMD ["--help"]