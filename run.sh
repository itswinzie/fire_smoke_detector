#!/bin/bash
# =============================================================================
#  Fire & Smoke Detection System — Setup & Launch
#  Jetson Orin NX | Ollama gemma3:4b
# =============================================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo ""
echo "============================================================"
echo "  🔥  FIRE & SMOKE DETECTION SYSTEM"
echo "       Jetson Orin NX | Ollama gemma3:4b"
echo "============================================================"

# ── 1. Ollama ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/3]  Checking Ollama..."
if ! command -v ollama &>/dev/null; then
  echo "       Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "       ✅  Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
fi

# Start Ollama service if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "       Starting Ollama service..."
  ollama serve &>/dev/null &
  sleep 4
fi

# ── 2. Pull model ─────────────────────────────────────────────────────────────
echo ""
echo "[2/3]  Checking model gemma3:4b..."
if ollama list 2>/dev/null | grep -q "gemma3:4b"; then
  echo "       ✅  gemma3:4b already downloaded"
else
  echo "       Downloading gemma3:4b (this may take several minutes)..."
  ollama pull gemma3:4b
  echo "       ✅  Download complete"
fi

# ── 3. Python dependencies ────────────────────────────────────────────────────
echo ""
echo "[3/3]  Installing Python dependencies..."
pip install --break-system-packages -q aiohttp opencv-python 2>/dev/null \
  || pip install -q aiohttp opencv-python
echo "       ✅  Dependencies ready"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Launching detection system..."
echo "============================================================"
echo ""
python3 detector.py
