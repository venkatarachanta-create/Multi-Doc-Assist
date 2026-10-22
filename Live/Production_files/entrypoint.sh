#!/bin/sh
# Starts both processes in a single container:
#   1. FastAPI (internal only, 127.0.0.1:8000) — the RAG backend
#   2. Streamlit (public, 0.0.0.0:$PORT)       — the UI everything else talks to
#
# $PORT is read at container start, defaulting to 8080 when nothing sets it —
# deliberately NOT 8000, since that's FastAPI's fixed internal port and 0.0.0.0
# binds collide with a 127.0.0.1 bind on the same port number:
#   - Local docker run          -> not set, falls back to 8080 (map with -p 8080:8080)
#   - Render                    -> sets $PORT automatically
#   - Azure Web App for Containers -> doesn't set $PORT itself; instead you tell Azure
#     which port the container listens on via the WEBSITES_PORT app setting. Set
#     WEBSITES_PORT=8080 to match the default here.
set -e

PORT="${PORT:-8080}"

echo "Starting FastAPI (internal, 127.0.0.1:8000)..."
uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --loop asyncio &

echo "Waiting for FastAPI to become healthy..."
until curl -s http://127.0.0.1:8000/health > /dev/null; do
    sleep 1
done

echo "Starting Streamlit (public, 0.0.0.0:${PORT})..."
exec python -m streamlit run src/frontend/app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true
