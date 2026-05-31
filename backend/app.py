from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR      = Path(__file__).parent
FRONTEND_DIR  = BASE_DIR.parent / "frontend"
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

pipeline: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    if PIPELINE_PATH.exists():
        pipeline = joblib.load(PIPELINE_PATH)
        acc = pipeline.get("test_accuracy", "n/a")
        print(f"Pipeline loaded — test accuracy: {acc}")
    else:
        print("No pipeline.pkl found. Run: .venv/Scripts/python backend/train_model.py")
    yield
    pipeline = None


app = FastAPI(title="TomatoSense", version="1.0.0", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static",
)
templates = Jinja2Templates(directory=FRONTEND_DIR / "templates")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    info: dict = {"status": "ok", "model_loaded": pipeline is not None}
    if pipeline:
        info["test_accuracy"]      = pipeline.get("test_accuracy")
        info["explained_variance"] = pipeline.get("explained_variance")
        info["kernel"]             = pipeline.get("kernel")
    return info


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
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. The file may be corrupted.",
        )

    img_size = pipeline["img_size"]
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    X = img_resized.astype(np.float32) / 255.0
    X = X.reshape(1, -1)

    X_scaled = pipeline["scaler"].transform(X)
    X_pca    = pipeline["pca"].transform(X_scaled)

    label_idx = int(pipeline["model"].predict(X_pca)[0])
    proba     = pipeline["model"].predict_proba(X_pca)[0]
    classes   = pipeline["classes"]

    return {
        "label":       classes[label_idx],
        "label_index": label_idx,
        "confidence":  round(float(proba[label_idx]) * 100, 2),
        "probabilities": {
            classes[0]: round(float(proba[0]) * 100, 2),
            classes[1]: round(float(proba[1]) * 100, 2),
        },
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/classify")
def classify_page(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})


@app.get("/analytics")
def analytics_page(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})


@app.get("/about")
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})
