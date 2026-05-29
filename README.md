# 🌿 PlantDoc AI — Botanical Disease Diagnostic System

> **A full-stack, production-ready AI system** for real-time plant disease detection using deep learning (MobileNetV2 Transfer Learning) trained on the PlantVillage dataset across **38 plant disease classes**.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Project Structure](#-project-structure)
4. [Tech Stack](#-tech-stack)
5. [Dataset](#-dataset)
6. [Supported Disease Classes (38)](#-supported-disease-classes-38)
7. [ML Pipeline](#-ml-pipeline)
   - [Model Architecture](#model-architecture)
   - [Training Strategy](#training-strategy)
   - [Data Augmentation](#data-augmentation)
8. [Backend API](#-backend-api)
9. [Frontend Web App](#-frontend-web-app)
10. [Performance Metrics](#-performance-metrics)
11. [Setup & Installation](#-setup--installation)
12. [Step-by-Step Usage Guide](#-step-by-step-usage-guide)
13. [API Reference](#-api-reference)
14. [Output Artifacts](#-output-artifacts)
15. [Troubleshooting](#-troubleshooting)

---

## 🌱 Project Overview

**PlantDoc AI** is an end-to-end intelligent plant pathology diagnostic platform. A user uploads a leaf photograph, and the system:

1. Runs the image through a fine-tuned **MobileNetV2** CNN.
2. Returns a diagnosis with confidence score, plant name, and disease label.
3. Displays a rich **interactive report** in the web portal with treatment protocols, severity classification, environmental risk factors, and recovery steps.
4. Allows the report to be **downloaded as a PDF** or **printed**.

The platform is designed for farmers, agronomists, and botanical researchers who need instant, reliable disease identification without specialized lab equipment.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PlantDoc AI System                           │
├──────────────────────┬──────────────────────┬───────────────────────┤
│   ML Pipeline        │   FastAPI Backend     │   React Frontend      │
│   (Python)           │   (Python)            │   (Vite + Tailwind)   │
│                      │                       │                       │
│  PlantVillage        │  POST /predict        │  Home Screen          │
│  Dataset (38cls)     │  ─────────────────    │  Diagnose Screen      │
│       │              │  • CORS enabled       │  Results Screen       │
│  prepare_test_set.py │  • File upload        │  The Engine Screen    │
│       │              │  • Inference engine   │                       │
│  train.py            │  • Auto image cleanup │  PDF Report Export    │
│  (MobileNetV2)       │  • Static file serve  │  Print Report         │
│       │              │                       │                       │
│  evaluate.py         │  GET /               │  API: localhost:8000  │
│  (Confusion Matrix)  │  ─────────────────    │  UI: localhost:5173   │
│       │              │  • Health check       │                       │
│  predict.py          │                       │                       │
│  (Inference CLI)     │                       │                       │
│                      │                       │                       │
│  models/             │  api/uploads/         │  web/src/             │
│  best_model.keras    │  (temp image store)   │  App.jsx              │
│  class_indices.json  │                       │  index.css            │
└──────────────────────┴──────────────────────┴───────────────────────┘
```

---

## 📁 Project Structure

```
PLANT_DOC/
├── Plant dataset/              # PlantVillage dataset (not tracked by git)
│   ├── train/                  # ~80% of images — 38 class folders
│   ├── val/                    # ~10% of images — 38 class folders
│   └── test/                   # ~10% of images — created by prepare_test_set.py
│
├── models/                     # Saved model artifacts
│   ├── best_model.keras        # Trained MobileNetV2 model (~31 MB)
│   └── class_indices.json      # Class name → index mapping (38 classes)
│
├── outputs/                    # Evaluation results
│   ├── confusion_matrix.png    # 38×38 normalized confusion matrix heatmap
│   └── classification_report.txt  # Per-class precision / recall / F1
│
├── api/                        # FastAPI backend
│   ├── app.py                  # Main API: /predict endpoint + CORS + cleanup
│   └── uploads/                # Temporary uploaded image storage (24h TTL)
│
├── web/                        # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx             # All screens: Home, Diagnose, Results, Engine
│   │   ├── index.css           # Global design system (Tailwind v4)
│   │   ├── main.jsx            # React entry point
│   │   ├── components/ui/      # Shared UI components
│   │   └── lib/                # Utility functions
│   ├── public/                 # Static assets (specimen_hero.png, advisor.png)
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── utils/
│   └── dataset_utils.py        # Dataset verification & class consistency checks
│
├── prepare_test_set.py         # Carves test split from val/ (run once)
├── train.py                    # Two-phase transfer learning + fine-tuning
├── evaluate.py                 # Full evaluation on test set
├── predict.py                  # CLI inference tool (auto or manual image paths)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🛠 Tech Stack

### Machine Learning / Backend
| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow ≥ 2.13 / Keras |
| Base Model | MobileNetV2 (ImageNet pre-trained) |
| Image Processing | Pillow (PIL) |
| Numerical Computing | NumPy |
| Evaluation Metrics | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Progress Bars | tqdm |
| API Framework | FastAPI |
| API Server | Uvicorn |
| File Uploads | python-multipart |

### Frontend
| Component | Technology |
|-----------|-----------|
| Framework | React 19 |
| Build Tool | Vite 8 |
| Styling | Tailwind CSS v4 |
| Animations | Framer Motion |
| Icons | Lucide React |
| PDF Generation | jsPDF |
| CSS Utilities | clsx, tailwind-merge |

---

## 📊 Dataset

**PlantVillage Dataset** — The gold standard benchmark for plant disease recognition.

| Property | Detail |
|----------|--------|
| Total Classes | 38 (plant + disease combinations) |
| Dataset Split | train (~80%) / val (~10%) / test (~10%) |
| Image Format | JPEG (.jpg / .JPG) |
| Image Source | Controlled lab conditions |
| Unique Plants | 14 plant species |
| Healthy Classes | 14 (one per plant species) |
| Disease Classes | 24 |
| Samples per class (training cap) | 150 (configurable in train.py) |

> ⚠️ The dataset is **not included** in the repository. Download it from [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and place it as `Plant dataset/` in the project root.

---

## 🌾 Supported Disease Classes (38)

| # | Class Label | Plant | Condition |
|---|-------------|-------|-----------|
| 0 | Apple___Apple_scab | Apple | Disease |
| 1 | Apple___Black_rot | Apple | Disease |
| 2 | Apple___Cedar_apple_rust | Apple | Disease |
| 3 | Apple___healthy | Apple | Healthy |
| 4 | Blueberry___healthy | Blueberry | Healthy |
| 5 | Cherry_(including_sour)___Powdery_mildew | Cherry | Disease |
| 6 | Cherry_(including_sour)___healthy | Cherry | Healthy |
| 7 | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn | Disease |
| 8 | Corn_(maize)___Common_rust_ | Corn | Disease |
| 9 | Corn_(maize)___Northern_Leaf_Blight | Corn | Disease |
| 10 | Corn_(maize)___healthy | Corn | Healthy |
| 11 | Grape___Black_rot | Grape | Disease |
| 12 | Grape___Esca_(Black_Measles) | Grape | Disease |
| 13 | Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | Grape | Disease |
| 14 | Grape___healthy | Grape | Healthy |
| 15 | Orange___Haunglongbing_(Citrus_greening) | Orange | Disease |
| 16 | Peach___Bacterial_spot | Peach | Disease |
| 17 | Peach___healthy | Peach | Healthy |
| 18 | Pepper,_bell___Bacterial_spot | Bell Pepper | Disease |
| 19 | Pepper,_bell___healthy | Bell Pepper | Healthy |
| 20 | Potato___Early_blight | Potato | Disease |
| 21 | Potato___Late_blight | Potato | Disease |
| 22 | Potato___healthy | Potato | Healthy |
| 23 | Raspberry___healthy | Raspberry | Healthy |
| 24 | Soybean___healthy | Soybean | Healthy |
| 25 | Squash___Powdery_mildew | Squash | Disease |
| 26 | Strawberry___Leaf_scorch | Strawberry | Disease |
| 27 | Strawberry___healthy | Strawberry | Healthy |
| 28 | Tomato___Bacterial_spot | Tomato | Disease |
| 29 | Tomato___Early_blight | Tomato | Disease |
| 30 | Tomato___Late_blight | Tomato | Disease |
| 31 | Tomato___Leaf_Mold | Tomato | Disease |
| 32 | Tomato___Septoria_leaf_spot | Tomato | Disease |
| 33 | Tomato___Spider_mites Two-spotted_spider_mite | Tomato | Disease |
| 34 | Tomato___Target_Spot | Tomato | Disease |
| 35 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato | Disease |
| 36 | Tomato___Tomato_mosaic_virus | Tomato | Disease |
| 37 | Tomato___healthy | Tomato | Healthy |

---

## 🧠 ML Pipeline

### Model Architecture

The model uses **MobileNetV2** as a feature extractor with a custom classification head.

```
Input (160×160×3)
        │
        ▼
┌───────────────────┐
│ Data Augmentation │  RandomFlip, RandomRotation(±20%), RandomZoom(20%)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   MobileNetV2     │  ImageNet weights (frozen in Phase 1)
│  (Feature Extractor) │  ~2.3M parameters
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ GlobalAvgPool2D   │  Spatial → vector
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Dropout (0.3)    │  Regularization
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Dense (38, softmax) │  Output: 38-class probability distribution
└───────────────────┘
```

| Hyperparameter | Value |
|---------------|-------|
| Input Shape | 160 × 160 × 3 |
| Batch Size | 16 |
| Phase 1 Epochs | 10 (with EarlyStopping, patience=3) |
| Phase 2 Epochs | Continued from Phase 1 (patience=5) |
| Phase 1 Optimizer | Adam (default lr=1e-3) |
| Phase 2 Optimizer | Adam (lr=1e-5) |
| Phase 2 Loss | CategoricalCrossentropy(label_smoothing=0.1) |
| Fine-tuned layers | Last 50 layers of MobileNetV2 |
| Samples per class | 150 (to cap training time) |
| Model checkpoint | `save_best_only=True` |

### Training Strategy

Training is performed in **two phases** for optimal accuracy:

**Phase 1 — Top-layer training:**
- MobileNetV2 base is fully **frozen**.
- Only the custom classification head is trained.
- Uses standard Adam optimizer.
- Trains until early stopping triggers (patience=3).

**Phase 2 — Fine-tuning:**
- The **last 50 layers** of MobileNetV2 are **unfrozen**.
- Very low learning rate (1e-5) prevents catastrophic forgetting.
- Label smoothing (0.1) improves generalization.
- Trains until early stopping triggers (patience=5).
- Best model is continuously checkpointed.

### Data Augmentation

Applied **inline via Keras layers** (only active during training):

| Transform | Value |
|-----------|-------|
| Horizontal + Vertical Flip | Random |
| Rotation | ±20° |
| Zoom | ±20% |

Preprocessing: `mobilenet_v2.preprocess_input` (scales pixels to [-1, 1]).

---

## 🚀 Backend API

**Framework:** FastAPI | **Server:** Uvicorn | **Port:** 8000

### Endpoints

#### `GET /`
Health check. Returns model name and online status.
```json
{ "status": "online", "auth": "disabled", "model": "model_name" }
```

#### `POST /predict`
Upload a leaf image and receive a full diagnostic response.

**Request:** `multipart/form-data` — field: `file` (image file)

**Response:**
```json
{
  "plant": "Tomato",
  "disease": "Late blight",
  "is_healthy": false,
  "confidence": 0.9423,
  "filename": "leaf.jpg",
  "img_url": "/uploads/<uuid>_leaf.jpg",
  "timestamp": "2026-05-02T07:50:00",
  "specimen_id": "PD-A3F91C"
}
```

### Key Features
- **Automatic preprocessing**: EXIF rotation correction, resize to model input shape, MobileNetV2 normalization.
- **Smart label parsing**: Converts `Tomato___Late_blight` → `{ plant: "Tomato", disease: "Late blight" }`.
- **Background cleanup**: Uploaded images older than 24 hours are automatically deleted on each request.
- **Static file serving**: Uploaded images accessible at `/uploads/<filename>`.
- **CORS**: Whitelisted for `http://localhost:5173` and `http://127.0.0.1:5173`.

### Running the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Or directly:
```bash
python api/app.py
```

---

## 🖥 Frontend Web App

**Framework:** React 19 + Vite 8 | **Styling:** Tailwind CSS v4 | **Port:** 5173

### Screens / Pages

| Screen | Route (State) | Description |
|--------|--------------|-------------|
| Home | `screen='home'` | Hero banner, accuracy stats, CTA |
| Diagnose | `screen='diagnose'` | Drag-and-drop / file upload |
| Results | `screen='results'` | Full diagnostic report |
| The Engine | `screen='engine'` | Model architecture info |

### Results Screen Features

After a successful prediction, the Results Screen displays:

- 🔍 **Specimen image** with animated disease marker overlay
- 📊 **Confidence Score** — animated progress bar (0–100%)
- ⚠️ **Severity Classification** — Low Risk / Caution / Critical (color-coded)
- 🦠 **Possible Causes** — tag chips for environmental pathogenesis factors
- 🌡 **Environmental Risk Panel** — optimal humidity & temperature for disease spread
- 📋 **Care Guide Sidebar** (sticky):
  - Diagnosis Verdict (clinical description)
  - Organic treatment recommendation
  - Chemical protocol
  - Numbered execution/recovery steps
- 💊 **Pathologist Tip** — contextual advice (healthy vs. diseased)

### Export Options

| Feature | Description |
|---------|-------------|
| **Download PDF** | Generates a formal clinical report via jsPDF, including specimen image, all metrics, treatment plan, and digital signature footer |
| **Print** | Browser print with a formatted print-only layout (`PrintReport` component) |

### Running the Frontend

```bash
cd web
npm install
npm run dev
```

Visit **http://localhost:5173**

---

## 📈 Performance Metrics

| Metric | Expected (PlantVillage benchmark) | Achieved |
|--------|-----------------------------------|---------|
| Validation Accuracy | 94–97% | **88.2%** |
| Test Accuracy | 93–96% | **85.85%** |

> ℹ️ The gap from benchmark is expected: the `SAMPLES_PER_CLASS = 150` cap limits training data to ~10% of full PlantVillage to reduce training time. Remove or raise this cap for higher accuracy.

**Evaluation outputs** (saved to `outputs/`):
- `confusion_matrix.png` — Normalized 38×38 heatmap (Blues colormap)
- `classification_report.txt` — Per-class precision, recall, F1-score, support

---

## ⚙️ Setup & Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.8 – 3.11 |
| Node.js | 18+ |
| npm | 9+ |
| GPU (optional) | CUDA-compatible (strongly recommended for training) |

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PLANT_DOC
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` includes:**
```
tensorflow>=2.13.0
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
```

### 4. Install Frontend Dependencies

```bash
cd web
npm install
cd ..
```

### 5. Add the Dataset

Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and arrange it as:

```
PLANT_DOC/
└── Plant dataset/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___healthy/
    │   └── ... (38 folders)
    └── val/
        ├── Apple___Apple_scab/
        └── ... (38 folders)
```

---

## 📖 Step-by-Step Usage Guide

### Step 0: Verify Dataset Integrity

Checks all 38 classes, counts images, and detects corrupt files.

```bash
python utils/dataset_utils.py
```

Expected output:
- 38 classes found in `train/` and `val/`
- 0 corrupt files

---

### Step 1: Create Test Split *(run once)*

Moves **50%** of each `val/` class's images into a new `test/` folder. The script is **idempotent** — safe to re-run; existing test images are skipped.

```bash
python prepare_test_set.py
```

> ⚠️ **Back up your dataset first!** Images are *moved*, not copied.

**Result:**
- `val/` → ~10% of full dataset
- `test/` → ~10% of full dataset (new folder created)
- `train/` → unchanged

---

### Step 2: Train the Model

```bash
python train.py
```

**What happens:**
1. Loads `train/` and `val/` splits using `tf.data` pipeline.
2. Builds MobileNetV2 model with frozen base + augmentation head.
3. **Phase 1**: Trains classification head for up to 10 epochs.
4. **Phase 2**: Unfreezes last 50 MobileNetV2 layers and fine-tunes with lr=1e-5.
5. Saves best model → `models/best_model.keras`
6. Saves class mapping → `models/class_indices.json`

**Estimated time:** 10–30 minutes (GPU) | 2–4 hours (CPU)

---

### Step 3: Evaluate on Test Set

```bash
python evaluate.py
```

**Outputs:**
- Test accuracy & loss printed to console
- `outputs/confusion_matrix.png` — normalized heatmap
- `outputs/classification_report.txt` — per-class metrics

---

### Step 4: CLI Prediction

```bash
# Auto: picks 5 random images from test/ and evaluates them
python predict.py

# Manual: provide your own images
python predict.py path/to/leaf.jpg path/to/another.jpg
```

**Output includes:**
- Predicted class + confidence %
- Top-3 predictions with confidence bars
- Auto-evaluation against true label (if image is from test set)

---

### Step 5: Launch the Full Web Portal

**Terminal 1 — Start the API:**
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Start the Frontend:**
```bash
cd web
npm run dev
```

Visit **http://localhost:5173** to use the interactive portal.

---

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Upload an image and get diagnosis |
| `GET` | `/uploads/{filename}` | Serve an uploaded image |

### `POST /predict` — Full Details

**Request Headers:**
```
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | image file | ✅ Yes | JPG, PNG, BMP, JPEG leaf photo |

**Success Response (200):**
```json
{
  "plant": "Tomato",
  "disease": "Late blight",
  "is_healthy": false,
  "confidence": 0.9423,
  "filename": "leaf.jpg",
  "img_url": "/uploads/a1b2c3d4_leaf.jpg",
  "timestamp": "2026-05-02T07:50:00.000000",
  "specimen_id": "PD-A3F91C"
}
```

**Error Responses:**
| Status | Condition |
|--------|-----------|
| 400 | Empty file uploaded |
| 500 | Model inference failed |

---

## 📦 Output Artifacts

| File | Location | Description |
|------|----------|-------------|
| Trained model | `models/best_model.keras` | Best checkpoint from training (~31 MB) |
| Class mapping | `models/class_indices.json` | JSON: `{class_name: index}` for all 38 classes |
| Confusion matrix | `outputs/confusion_matrix.png` | Normalized 38×38 heatmap (150 DPI) |
| Classification report | `outputs/classification_report.txt` | Precision, recall, F1-score per class |
| Uploaded images | `api/uploads/` | Temporary storage; auto-deleted after 24h |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tensorflow` | Run `pip install -r requirements.txt` in your virtual environment |
| `FileNotFoundError: Plant dataset/test` | Run `python prepare_test_set.py` first |
| `FileNotFoundError: models/best_model.keras` | Run `python train.py` first |
| Out of memory during training | Lower `BATCH_SIZE` in `train.py` (try `8`) |
| Slow training on CPU | Lower `SAMPLES_PER_CLASS` to `50` for a smoke test |
| MobileNetV2 download fails | Check internet connection (first run only downloads ImageNet weights) |
| CORS error on frontend | Ensure API is running on port 8000 and frontend on port 5173 |
| `npm install` fails | Check Node.js version is 18+. Try `npm install --legacy-peer-deps` |
| Empty prediction / wrong class | Ensure `class_indices.json` matches the model's training class order |
| Vite build errors | Delete `web/node_modules/` and `web/dist/`, then re-run `npm install` |

---

## 🧪 Development Tips

- **Increase accuracy**: Remove the `SAMPLES_PER_CLASS = 150` cap in `train.py` to train on the full PlantVillage dataset.
- **Faster iteration**: Set `INITIAL_EPOCHS = 3` for a quick smoke test without full training.
- **Custom images**: Pass any leaf photo directly to `predict.py` for instant CLI diagnosis.
- **Dataset verification**: Run `utils/dataset_utils.py` after any dataset modification to catch issues early.
- **Model inspection**: Load `models/best_model.keras` in a Jupyter notebook with `model.summary()` to inspect layers.

---

## 📄 License

This project is intended for educational and research purposes. The PlantVillage dataset is used under its respective academic license.

---

<div align="center">

**PlantDoc AI** — Empowering Botanical Health through Deep Learning 🌿

*MobileNetV2 · FastAPI · React · TensorFlow · PlantVillage*

</div>
