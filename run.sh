#!/bin/bash
# ============================================================
#  Fire & Smoke Detection — Setup & Run Script
#  Jetson Orin NX | Ollama + gemma3:4b
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  🔥 FIRE & SMOKE DETECTION SYSTEM — SETUP"
echo "============================================================"

# ── 1. Check Ollama ──────────────────────────────────────────
echo ""
echo "[ 1/4 ] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
  echo "  ❌ Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "  ✅ Ollama found: $(ollama --version)"
fi

# Ensure Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  ▶  Starting Ollama service..."
  ollama serve &
  sleep 3
fi

# ── 2. Pull model ─────────────────────────────────────────────
echo ""
echo "[ 2/4 ] Checking model gemma3:4b..."
if ollama list | grep -q "gemma3:4b"; then
  echo "  ✅ gemma3:4b already downloaded"
else
  echo "  ⬇  Downloading gemma3:4b (this may take a while)..."
  ollama pull gemma3:4b
fi

# ── 3. Python dependencies ────────────────────────────────────
echo ""
echo "[ 3/4 ] Installing Python dependencies..."
pip install --break-system-packages -q aiohttp opencv-python 2>/dev/null || \
pip install -q aiohttp opencv-python
echo "  ✅ Dependencies ready"

# ── 4. Create logs directory ──────────────────────────────────
mkdir -p logs

# ── 5. Run ────────────────────────────────────────────────────
echo ""
echo "[ 4/4 ] Launching detection system..."
echo ""
python3 detector.py
