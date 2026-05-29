"""
api/app.py - Production AI API for Plant Pathology
Features: Efficient inference, Automated upload cleanup
"""
import io, json, os, logging, uuid
from datetime import datetime, timedelta
import time
from typing import Optional

import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# --- Config ---
BASE_DIR        = Path(__file__).resolve().parent.parent
MODEL_PATH      = str(BASE_DIR / "models" / "best_model.keras")
CLASS_IDX_PATH  = str(BASE_DIR / "models" / "class_indices.json")
UPLOAD_DIR      = BASE_DIR / "api" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- ML Model Load ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_model = tf.keras.models.load_model(MODEL_PATH)
_IMG_H, _IMG_W = _model.input_shape[1], _model.input_shape[2]
with open(CLASS_IDX_PATH) as f:
    _idx_to_class = {v: k for k, v in json.load(f).items()}

# --- Helpers ---
def preprocess(raw: bytes):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = ImageOps.exif_transpose(img) 
    img = img.resize((_IMG_W, _IMG_H), Image.BILINEAR)
    arr = np.array(img, dtype="float32")
    if "efficientnet" in _model.name.lower(): pass
    elif "mobilenet" in _model.name.lower():
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def parse_label(raw: str):
    parts = raw.split("___", 1)
    plant = parts[0].replace("_", " ")
    disease = (parts[1] if len(parts) > 1 else "Unknown").replace("_", " ").capitalize()
    return {"plant": plant, "disease": disease, "is_healthy": "healthy" in disease.lower()}

def cleanup_uploads():
    """Delete files older than 24 hours from the uploads directory."""
    try:
        now = time.time()
        cutoff = now - (24 * 3600)
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                log.info(f"Cleaned up old file: {f.name}")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")

# --- App ---
app = FastAPI(title="PlantDoc AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

@app.get("/")
def root(): return {"status": "online", "auth": "disabled", "model": _model.name}

# --- Diagnosis Endpoints ---
@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    background_tasks.add_task(cleanup_uploads)
    raw = await file.read()
    if not raw: raise HTTPException(400, "Empty file")
    
    try:
        tensor = preprocess(raw)
        probs  = _model.predict(tensor, verbose=0)[0]
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(500, str(e))

    idx = int(np.argsort(probs)[::-1][0])
    primary = {
        "confidence": float(probs[idx]), 
        **parse_label(_idx_to_class[idx])
    }

    img_filename = f"{uuid.uuid4()}_{file.filename}"
    img_path = UPLOAD_DIR / img_filename
    with open(img_path, "wb") as f: f.write(raw)

    doc = {
        **primary,
        "filename": file.filename,
        "img_url": f"/uploads/{img_filename}",
        "timestamp": datetime.utcnow().isoformat(),
        "specimen_id": f"PD-{uuid.uuid4().hex[:6].upper()}"
    }
    
    return doc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
