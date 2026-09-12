# TomatoSense

TomatoSense is a FastAPI web application that tells you whether a tomato is **ripe** or **unripe** from a photo. It pairs a small machine-learning pipeline with a simple, two-page browser UI aimed at everyday users.

Live app: https://tomato-sense.onrender.com

## What it does

- Upload, drop, or paste a tomato photo and get a ripe/unripe answer with a plain-language confidence score.
- See the photos you've checked during the current visit (kept in the browser only; nothing is stored server-side).
- Read a short "How it works" page with the model's accuracy in everyday terms and its limitations.

## How it is built

The application is split into three parts:

- **Backend**: FastAPI serves pages, JSON endpoints, and model inference.
- **Frontend**: Jinja2 templates, a small hand-written stylesheet, and vanilla JavaScript. No build step or CSS framework.
- **Model pipeline**: A serialized `joblib` pipeline is loaded from `backend/models/pipeline.pkl` at startup.

The model workflow in `backend/train_model.py` is:

1. Load tomato images from the Kaggle datasets under `notebook/data/tomato_kaggle/`.
2. Infer labels from filenames and parent folder names.
3. Resize images to `64 x 64`, convert to RGB, normalize to `[0, 1]`, and flatten to feature vectors.
4. Apply `StandardScaler`, then `PCA(100)`.
5. Train and compare linear, RBF, and polynomial SVM kernels.
6. Save the fitted scaler, PCA, model, and analytics metadata into `backend/models/pipeline.pkl`.

## Technology

- Python 3.11
- FastAPI
- Uvicorn
- Jinja2
- OpenCV
- NumPy
- scikit-learn
- Joblib
- Inter (Google Fonts)

## Repository structure

- `backend/` FastAPI application, training script, and saved model artifacts
- `frontend/templates/` the two pages (`index.html`, `about.html`), the shared shell (`base.html`) and inline SVG icons (`_icons.html`)
- `frontend/static/js/classify.js` upload, prediction, and result handling
- `frontend/static/css/style.css` all styling
- `notebook/` notebook and dataset files used during model development
- `render.yaml` Render deployment configuration

## Current functionality

- `GET /` the classifier: upload a photo and get a result
- `GET /about` plain-language "How it works" page with accuracy and limitations
- `POST /predict` image inference endpoint
- `GET /analytics-data` evaluation metrics as JSON (kernel comparison, confusion matrix, per-class metrics, dataset summary)
- `GET /stats` in-memory prediction counts for the current server session
- `GET /health` health check and model-load status
- `GET /classify` and `GET /analytics` redirect to `/` and `/about` (kept for old links)

Prediction accepts JPEG, PNG, WEBP, and BMP files up to 10 MB. If the serialized pipeline is missing, the app will start but prediction and analytics endpoints will return an error until `backend/train_model.py` is run.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Ensure the trained pipeline exists at `backend/models/pipeline.pkl`.
4. Start the app:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

5. Open `http://127.0.0.1:8000/` in a browser.

## Training the model

Run the training script from the project root:

```bash
.venv\Scripts\python backend\train_model.py
```

The script expects the dataset files to be present under `notebook/data/tomato_kaggle/`.

## Deployment

The repository includes a `render.yaml` file for deployment on Render. It installs the backend dependencies and starts the app with Uvicorn.

Deployed application: https://tomato-sense.onrender.com

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.