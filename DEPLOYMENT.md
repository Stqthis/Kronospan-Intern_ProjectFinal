# Deployment

## Local (no Docker)

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ollama serve &                       # plus: ollama pull qwen2.5-coder:32b
    python app.py                        # desktop GUI
    # or headless:
    python -m jarvisman.cli index ./my_files
    python -m jarvisman.cli chat

## Docker (headless)

    docker compose up -d ollama
    docker compose exec ollama ollama pull qwen2.5-coder:32b
    docker compose exec ollama ollama pull nomic-embed-text
    mkdir -p files                       # drop your PDFs/Excels here
    docker compose run --rm app index /files
    docker compose run --rm app ask "Total funds of Lignum Technologies AG in EUR?"
    docker compose run --rm app chat

The index persists in the `jarvis_index` volume; re-running `index` on new
files UPDATES the collection (previously indexed files are kept, re-indexed
files replace their old version). GUI and CLI share the same index format.

## Docker (GUI, Linux/X11)

    xhost +local:docker
    docker compose run --rm \
      -e DISPLAY=$DISPLAY -e QT_QPA_PLATFORM=xcb \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      --entrypoint python app app.py

On Windows/macOS use an X server (VcXsrv / XQuartz) and set DISPLAY
accordingly, or simply run the GUI natively and let only Ollama run in Docker
(`OLLAMA_HOST=http://localhost:11434` is the default).

## Scaling to many files

- Index in batches: `python -m jarvisman.cli index folderA folderB ...` --
  updates are incremental, and per-table LLM work (profiling, cards) is
  cached by schema hash, so unchanged files cost zero model calls on rebuild.
- Verify the foundation any time: `python -m jarvisman.cli check --data <folder>`
  runs the ground-truth use-case checks offline.
- For lower latency, route planning to a smaller model:
  `RAG_PLAN_MODEL=qwen2.5-coder:7b` (keep the 32b for synthesis).
