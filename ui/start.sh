#!/bin/bash
# CineMatch — launch both FastAPI backend + Vite dev server
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../rag_venv"

# Kill stale processes on our ports
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5173 | xargs kill -9 2>/dev/null || true

# Activate project venv
source "$VENV/bin/activate"

# Install UI backend deps if needed (skip if already present)
python -c "import fastapi, uvicorn, multipart" 2>/dev/null || pip install -q fastapi "uvicorn[standard]" python-multipart

# Install frontend deps if node_modules missing
cd "$SCRIPT_DIR"
if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║       CfYourMovie — AI Movie Discovery        ║"
echo "  ║  Backend  →  http://localhost:8000/api        ║"
echo "  ║  Frontend →  http://localhost:5173  (open me) ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Load API keys from .env file (never hardcode secrets)
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
else
  echo "Warning: .env not found. Copy .env.example → .env and fill in keys."
fi

# Start FastAPI in background
uvicorn app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Cleanup on Ctrl+C
trap "kill $BACKEND_PID 2>/dev/null" EXIT

# Start Vite dev server in foreground
npm run dev
