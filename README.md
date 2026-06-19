# TomatoSense

TomatoSense is a FastAPI web application for classifying tomato images as **ripe** or **unripe**. It combines a small machine-learning pipeline with a browser UI for upload-based prediction, live session stats, and model analytics.

## What it does

- Upload a tomato image and receive a ripe/unripe prediction with confidence scores.
- View live session activity, including recent predictions and counts by class.
- Inspect model analytics such as kernel comparison, confusion matrix, per-class metrics, and dataset summary.
- Review an about page that documents the preprocessing pipeline and training setup.

## How it is built

The application is split into three parts:

- **Backend**: FastAPI serves pages, JSON endpoints, and model inference.
- **Frontend**: Jinja2 templates plus vanilla JavaScript provide the dashboard, upload flow, and charts.
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
- Tailwind CSS via CDN
- Chart.js via CDN

## Repository structure

- `backend/` FastAPI application, training script, and saved model artifacts
- `frontend/templates/` HTML pages rendered by the backend
- `frontend/static/js/` client-side behavior for navigation, uploads, analytics, and dashboard stats
- `frontend/static/css/` small custom styles
- `notebook/` notebook and dataset files used during model development
- `render.yaml` Render deployment configuration

## Current functionality

- `GET /` dashboard with model summary cards and session activity
- `GET /classify` upload-and-predict page
- `POST /predict` image inference endpoint
- `GET /analytics` model analytics page
- `GET /analytics-data` analytics JSON for charts
- `GET /about` model card and dataset summary page
- `GET /stats` in-memory prediction stats for the current server session
- `GET /health` health check and model-load status

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

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for the full text.