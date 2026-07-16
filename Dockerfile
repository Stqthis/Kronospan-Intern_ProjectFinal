# Jarvisman -- offline document/data assistant.
# Headless by default (CLI); the PyQt6 GUI also works with X11 forwarding,
# see DEPLOYMENT.md.
FROM python:3.12-slim

# Qt runtime libraries (only needed for the GUI; small enough to keep in one
# image so the same image serves both modes) + curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1 \
        libx11-xcb1 libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 \
        libxcb-shape0 libxcb-xkb1 libxkbcommon-x11-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted index lives here; mount a volume to keep it across containers.
ENV RAG_DATA_DIR=/data
VOLUME ["/data"]

# Ollama runs as a sibling service (see docker-compose.yml).
ENV OLLAMA_HOST=http://ollama:11434

ENTRYPOINT ["python", "-m", "jarvisman.cli"]
CMD ["--help"]
