from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR      = Path(__file__).parent
FRONTEND_DIR  = BASE_DIR.parent / "frontend"
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.pkl"

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


@app.get("/health")
def health():
    info: dict = {"status": "ok", "model_loaded": pipeline is not None}
    if pipeline:
        info["test_accuracy"]     = pipeline.get("test_accuracy")
        info["explained_variance"] = pipeline.get("explained_variance")
        info["kernel"]            = pipeline.get("kernel")
    return info


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
