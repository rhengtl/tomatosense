from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR      = Path(__file__).parent
FRONTEND_DIR  = BASE_DIR.parent / "frontend"
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"

ALLOWED_TYPES    = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

pipeline: dict | None = None
predictions_log: deque = deque(maxlen=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        acc = pipeline.get("test_accuracy", "n/a")
        print(f"Pipeline loaded — test accuracy: {acc}")
    else:
        print("No pipeline.pkl found. Run: .venv/Scripts/python backend/train_model.py")
    _sync_template_globals()
    yield
    pipeline = None
    _sync_template_globals()


def _sync_template_globals() -> None:
    """Expose model status to every template (used by the app shell)."""
    templates.env.globals["model_loaded"] = pipeline is not None


app = FastAPI(title="TomatoSense", version="1.0.0", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")
templates.env.globals["current_year"] = datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.api_route("/health", methods=["GET", "HEAD"])  # HEAD: uptime monitors default to it
def health():
    info: dict = {"status": "ok", "model_loaded": pipeline is not None}
    if pipeline:
        info["test_accuracy"]      = pipeline.get("test_accuracy")
        info["explained_variance"] = pipeline.get("explained_variance")
        info["kernel"]             = pipeline.get("kernel")
    return info


@app.get("/analytics-data")
def analytics_data():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return pipeline["analytics"]


@app.get("/stats")
def stats():
    log = list(predictions_log)
    ripe_count   = sum(1 for p in log if p["label_index"] == 1)
    unripe_count = sum(1 for p in log if p["label_index"] == 0)
    recent = [
        {
            "label":      p["label"],
            "label_index": p["label_index"],
            "confidence": p["confidence"],
            "timestamp":  p["timestamp"],
        }
        for p in reversed(log)
    ][:5]
    return {
        "total":        len(log),
        "ripe_count":   ripe_count,
        "unripe_count": unripe_count,
        "recent":       recent,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run backend/train_model.py first.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG, PNG, WEBP, or BMP image.",
        )

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. The file may be corrupted.",
        )

    img_size    = pipeline["img_size"]
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    X        = img_resized.astype(np.float32) / 255.0
    X        = X.reshape(1, -1)
    X_scaled = pipeline["scaler"].transform(X)
    X_pca    = pipeline["pca"].transform(X_scaled)

    label_idx = int(pipeline["model"].predict(X_pca)[0])
    proba     = pipeline["model"].predict_proba(X_pca)[0]
    classes   = pipeline["classes"]

    result = {
        "label":       classes[label_idx],
        "label_index": label_idx,
        "confidence":  round(float(proba[label_idx]) * 100, 2),
        "probabilities": {
            classes[0]: round(float(proba[0]) * 100, 2),
            classes[1]: round(float(proba[1]) * 100, 2),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    predictions_log.append(result)
    return result


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/classify", include_in_schema=False)
def classify_redirect():
    # The classifier now lives on the home page; keep old links working.
    return RedirectResponse("/", status_code=301)


@app.get("/analytics", include_in_schema=False)
def analytics_redirect():
    # Detailed evaluation charts were folded into the plain-language "How it works" page.
    return RedirectResponse("/about", status_code=301)


@app.get("/about")
def about_page(request: Request):
    about_info = None
    if pipeline:
        cm = pipeline["analytics"]["confusion_matrix"]
        total   = sum(sum(row) for row in cm)
        correct = cm[0][0] + cm[1][1]
        about_info = {
            "accuracy":     f"{pipeline['test_accuracy'] * 100:.0f}%",
            "test_total":   total,
            "test_correct": correct,
        }
    return templates.TemplateResponse(
        "about.html", {"request": request, "about_info": about_info}
    )
