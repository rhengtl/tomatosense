"""
Train the tomato ripeness SVM pipeline and save it to backend/models/pipeline.pkl.

Mirrors the notebook pipeline exactly:
  - 64x64 RGB images, normalized to [0, 1], flattened to 12,288 features
  - Label inferred from filename / parent folder (ripe=1, unripe=0)
  - Capped at 200 images per class
  - StandardScaler -> PCA(100 components) -> SVC(kernel='linear', probability=True)
  - All three kernels evaluated; 20-trial holdout test stored in pipeline for analytics

Run from the project root:
    .venv/Scripts/python backend/train_model.py
"""

import os
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT  = Path(__file__).parent.parent
DATASET_DIR   = PROJECT_ROOT / "notebook" / "data" / "tomato_kaggle"
MODELS_DIR    = Path(__file__).parent / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"

IMG_SIZE      = 64
MAX_PER_CLASS = 200
N_COMPONENTS  = 100
CLASSES       = {0: "Unripe", 1: "Ripe"}
KERNELS       = ["linear", "rbf", "poly"]


def infer_label(path: str) -> int | None:
    p = Path(path)
    name = p.stem.lower().replace("-", " ").replace("_", " ")
    if any(x in name for x in ("unriped", "unripe", "green")):
        return 0
    if any(x in name for x in ("riped", "ripe", "red")):
        return 1
    parents = " ".join(
        part.lower().replace("-", " ").replace("_", " ") for part in p.parts[-3:-1]
    )
    if any(x in parents for x in ("unriped", "unripe", "green")):
        return 0
    if any(x in parents for x in ("riped", "ripe", "red")):
        return 1
    return None


def load_images(base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files: list[tuple[str, int]] = []

    for root, _, fnames in os.walk(base_dir):
        for fname in fnames:
            if fname.lower().endswith(exts):
                full = os.path.join(root, fname)
                label = infer_label(full)
                if label is not None:
                    files.append((full, label))

    rng = np.random.default_rng(42)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for cls in (0, 1):
        pool = [p for p, l in files if l == cls]
        keep = min(MAX_PER_CLASS, len(pool))
        chosen = rng.choice(len(pool), size=keep, replace=False)
        loaded = 0
        for idx in chosen:
            img = cv2.imread(pool[idx])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            images.append(img.astype(np.float32) / 255.0)
            labels.append(cls)
            loaded += 1
        print(f"  {CLASSES[cls]}: {loaded} images (pool {len(pool)})")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_DIR}\n"
            "Run the notebook first to download and extract the Kaggle datasets."
        )

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading images from:\n  {DATASET_DIR}\n")
    images, labels = load_images(DATASET_DIR)
    total = len(images)
    print(f"\nTotal: {total} — Ripe: {(labels==1).sum()}, Unripe: {(labels==0).sum()}")

    X = images.reshape(total, -1)

    # ── 80/20 split for kernel comparison and final evaluation ──────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.20, random_state=42, stratify=labels
    )

    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Fitting PCA ({N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"  Variance explained: {explained:.1f}%")

    # ── Evaluate all three kernels ──────────────────────────────────────────
    print("\nEvaluating all kernels...")
    kernel_accuracies: dict[str, float] = {}
    for k in KERNELS:
        clf = SVC(kernel=k, C=1.0, gamma="scale", random_state=42, probability=True)
        clf.fit(X_train_pca, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_pca))
        kernel_accuracies[k] = round(acc * 100, 2)
        print(f"  {k.upper():8} {acc*100:.2f}%")

    best_kernel = max(kernel_accuracies, key=kernel_accuracies.__getitem__)
    print(f"\nBest kernel: {best_kernel.upper()} ({kernel_accuracies[best_kernel]}%)")

    # ── Train final model on best kernel ────────────────────────────────────
    svm = SVC(kernel=best_kernel, C=1.0, gamma="scale", random_state=42, probability=True)
    svm.fit(X_train_pca, y_train)
    y_pred   = svm.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, target_names=["Unripe", "Ripe"], digits=4, output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=["Unripe", "Ripe"], digits=4))

    cm = confusion_matrix(y_test, y_pred).tolist()

    # ── 20-trial holdout evaluation ─────────────────────────────────────────
    print("Running 20-trial holdout evaluation...")
    X_tr60, X_unused, y_tr60, y_unused = train_test_split(
        X, labels, test_size=0.40, random_state=42, stratify=labels
    )
    sc2 = StandardScaler()
    pc2 = PCA(n_components=N_COMPONENTS, random_state=42)
    X_tr60_pca    = pc2.fit_transform(sc2.fit_transform(X_tr60))
    X_unused_pca  = pc2.transform(sc2.transform(X_unused))

    mdl2 = SVC(kernel=best_kernel, C=1.0, gamma="scale", random_state=42, probability=True)
    mdl2.fit(X_tr60_pca, y_tr60)

    trial_accuracies: list[float] = []
    for chunk in np.array_split(np.arange(len(X_unused_pca)), 20):
        if len(chunk) == 0:
            continue
        a = accuracy_score(y_unused[chunk], mdl2.predict(X_unused_pca[chunk]))
        trial_accuracies.append(round(a * 100, 2))

    trial_arr = np.array(trial_accuracies)
    print(f"  20-trial mean: {trial_arr.mean():.2f}%  std: {trial_arr.std():.2f}%")

    # ── Assemble and save pipeline ──────────────────────────────────────────
    pipeline = {
        # Inference objects
        "scaler":             scaler,
        "pca":                pca,
        "model":              svm,
        "img_size":           IMG_SIZE,
        "classes":            CLASSES,
        # Metadata
        "kernel":             best_kernel,
        "n_components":       N_COMPONENTS,
        "explained_variance": round(explained, 2),
        "test_accuracy":      round(accuracy, 4),
        # Analytics data (replaces hardcoded constant in app.py)
        "analytics": {
            "kernel_comparison": {
                "kernels":    [k.capitalize() if k != "rbf" else "RBF" for k in KERNELS],
                "accuracies": [kernel_accuracies[k] for k in KERNELS],
                "best":       best_kernel.capitalize() if best_kernel != "rbf" else "RBF",
            },
            "classification_report": {
                "Unripe":   {
                    "precision": round(report["Unripe"]["precision"], 4),
                    "recall":    round(report["Unripe"]["recall"],    4),
                    "f1":        round(report["Unripe"]["f1-score"],  4),
                    "support":   int(report["Unripe"]["support"]),
                },
                "Ripe": {
                    "precision": round(report["Ripe"]["precision"], 4),
                    "recall":    round(report["Ripe"]["recall"],    4),
                    "f1":        round(report["Ripe"]["f1-score"],  4),
                    "support":   int(report["Ripe"]["support"]),
                },
                "accuracy": round(accuracy, 4),
            },
            "confusion_matrix": cm,
            "trials": {
                "accuracies": trial_accuracies,
                "mean":       round(float(trial_arr.mean()), 2),
                "std":        round(float(trial_arr.std()),  2),
                "min":        round(float(trial_arr.min()),  2),
                "max":        round(float(trial_arr.max()),  2),
            },
            "dataset": {
                "total":              total,
                "ripe":               int((labels == 1).sum()),
                "unripe":             int((labels == 0).sum()),
                "pca_components":     N_COMPONENTS,
                "explained_variance": round(explained, 1),
                "img_size":           IMG_SIZE,
                "sources": [
                    "sumn2u/riped-and-unriped-tomato-dataset",
                    "nexuswho/tomatofruits",
                ],
            },
        },
    }

    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"\nPipeline saved to:\n  {PIPELINE_PATH}")


if __name__ == "__main__":
    main()
