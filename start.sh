#!/bin/bash
# ShotLab backend startup script
# Run from the bball/ root directory: ./start.sh

set -e
cd "$(dirname "$0")/backend"

echo ""
echo "  🏀  ShotLab Backend"
echo "  ─────────────────────────────────"

# Install deps if not present
if ! python3 -c "import fastapi, mediapipe, cv2" 2>/dev/null; then
  echo "  Installing dependencies..."
  pip install -r requirements.txt --break-system-packages -q
fi

echo "  Starting API on http://localhost:8000"
echo "  Open index.html in your browser, then click RUN ANALYSIS"
echo "  Press Ctrl+C to stop"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
