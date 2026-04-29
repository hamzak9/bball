"""
main.py — ShotLab FastAPI backend

Endpoints:
  POST /analyze   — upload 1–3 video files, returns analysis JSON
  GET  /health    — health check
  GET  /          — API info

Run locally:
  cd backend && uvicorn main:app --reload --port 8000
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analyzer.pipeline import run_analysis

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ShotLab API",
    description="AI basketball shooting analysis — pose + ball tracking",
    version="0.1.0",
)

# Allow the frontend (file:// or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_FILE_SIZE_MB   = 500


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "ShotLab API",
        "version": "0.1.0",
        "status":  "running",
        "endpoints": {
            "POST /analyze": "Upload 1–3 video files (front, side, rear), returns analysis",
            "GET /health":   "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    front: Optional[UploadFile] = File(default=None),
    side:  Optional[UploadFile] = File(default=None),
    rear:  Optional[UploadFile] = File(default=None),
):
    """
    Analyze 1–3 shooting videos.

    Accepts multipart form data with optional fields: front, side, rear.
    At least one must be provided.

    Returns structured JSON with:
      - n_shots:  number of shot arcs detected
      - overall:  0–100 composite score
      - arc, drift, release, balance, pocket: per-metric scores + details
      - per_shot: per-shot breakdown array
    """
    uploads = {"front": front, "side": side, "rear": rear}
    provided = {k: v for k, v in uploads.items() if v is not None}

    if not provided:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one video file (front, side, or rear).",
        )

    # Validate file types
    for angle, upload in provided.items():
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}' for {angle}. Use: {', '.join(ALLOWED_EXTENSIONS)}",
            )

    # Save uploads to temp files
    tmp_dir = tempfile.mkdtemp(prefix="shotlab_")
    video_paths: dict[str, str] = {}

    try:
        for angle, upload in provided.items():
            ext = Path(upload.filename or "video.mp4").suffix.lower()
            tmp_path = os.path.join(tmp_dir, f"{angle}{ext}")
            with open(tmp_path, "wb") as f:
                content = await upload.read()
                size_mb = len(content) / (1024 * 1024)
                if size_mb > MAX_FILE_SIZE_MB:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{angle} video is {size_mb:.0f}MB — max {MAX_FILE_SIZE_MB}MB",
                    )
                f.write(content)
            video_paths[angle] = tmp_path
            logger.info(f"Saved {angle} video: {tmp_path} ({size_mb:.1f}MB)")

        # Run analysis
        logger.info(f"Starting analysis for angles: {list(video_paths.keys())}")
        result = run_analysis(video_paths)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        # Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp dir: {tmp_dir}")
