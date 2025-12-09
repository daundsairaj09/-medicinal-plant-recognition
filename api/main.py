# api/main.py

import os
import json
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import numpy as np
# Import TensorFlow lazily — if it's not available the API will run in demo mode.
try:
    from tensorflow.keras.models import load_model as _load_model
    HAS_TF = True
except Exception:
    _load_model = None
    HAS_TF = False

from src.config import MODEL_DIR, IMG_SIZE, KNOWLEDGE_BASE

app = FastAPI(title="Medicinal Plant Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files under /static so the backend can host the web UI
try:
    from fastapi.staticfiles import StaticFiles
    STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'web'))
    if os.path.isdir(STATIC_DIR):
        app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
        print(f"\u2705 Mounted static files from {STATIC_DIR} at /static")
    else:
        print(f"\u26a0\ufe0f Static directory not found: {STATIC_DIR}")
except Exception as e:
    print(f"\u26a0\ufe0f Could not mount static files: {e}")

# ---------- Load model & metadata at startup (optional) ----------

MODEL_NAME = "mobilenet_v2.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# Do not fail startup if model or indices are missing — run in demo mode instead.
model = None
class_indices = {}
idx_to_class = {}

try:
    if HAS_TF:
        if os.path.exists(MODEL_PATH):
            model = _load_model(MODEL_PATH)
            print(f"✅ Loaded model from {MODEL_PATH}")
        else:
            print(f"⚠️ Model file not found at {MODEL_PATH}. API will run in demo mode.")
    else:
        print("⚠️ TensorFlow not available. API will run in demo mode.")

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"✅ Loaded class indices ({len(class_indices)} classes).")
    else:
        print(f"⚠️ class_indices.json not found at {CLASS_INDICES_PATH}.")
except Exception as e:
    print(f"⚠️ Failed loading model or class indices: {e}. Running in demo mode.")
    model = None
    class_indices = {}
    idx_to_class = {}

# Load medicinal uses knowledge base (optional)
plant_uses = {}
if os.path.exists(KNOWLEDGE_BASE):
    try:
        with open(KNOWLEDGE_BASE, "r") as f:
            content = f.read().strip()
            if content:
                plant_uses = json.loads(content)
                print(f"✅ Loaded plant uses for {len(plant_uses)} plants.")
            else:
                print("⚠️ plant_uses.json is empty. API will return empty info for plants.")
    except json.JSONDecodeError as e:
        print(f"⚠️ plant_uses.json is invalid JSON ({e}). API will return empty info for plants.")
else:
    print("⚠️ plant_uses.json not found, API will return empty info for plants.")


# ---------- Helper functions ----------

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and scale image same as training.
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_plant(img_array: np.ndarray):
    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    class_name = idx_to_class[idx]
    return class_name, confidence, preds.tolist()


# ---------- API endpoints ----------

@app.get("/")
def root():
    # If frontend index.html exists in the mounted static directory, serve it.
    try:
        index_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'web', 'index.html')
        index_path = os.path.abspath(index_path)
        if os.path.exists(index_path):
            return FileResponse(index_path, media_type='text/html')
    except Exception:
        pass

    return {"message": "Medicinal Plant Recognition API is running."}


# Backwards-compatible routes: serve top-level style/script if requested
@app.get("/style.css")
def serve_style():
    static_css = os.path.join(os.path.dirname(__file__), '..', 'app', 'web', 'style.css')
    static_css = os.path.abspath(static_css)
    if os.path.exists(static_css):
        return FileResponse(static_css, media_type='text/css')
    raise HTTPException(status_code=404, detail='style.css not found')


@app.get("/script.js")
def serve_script():
    static_js = os.path.join(os.path.dirname(__file__), '..', 'app', 'web', 'script.js')
    static_js = os.path.abspath(static_js)
    if os.path.exists(static_js):
        return FileResponse(static_js, media_type='application/javascript')
    raise HTTPException(status_code=404, detail='script.js not found')


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    content = await file.read()

    try:
        image = Image.open(BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # If model not loaded, return 503 so frontend can fallback to demo behavior.
    if model is None:
        return JSONResponse({
            "detail": "Model not available. API running in demo mode."}, status_code=503)

    img_array = preprocess_image(image)
    class_name, confidence, prob_list = predict_plant(img_array)

    info = plant_uses.get(class_name, {})
    response = {
        "plant_id": class_name,
        "confidence": confidence,
        "probabilities": prob_list,
        "info": info
    }

    return JSONResponse(response)
