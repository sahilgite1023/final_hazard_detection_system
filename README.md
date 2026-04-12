# Real-Time Hazard Sound Detection System

This project detects hazardous sounds from uploaded audio and microphone input using a Streamlit app.

## What this project contains

- `app.py`: Streamlit interface for upload + live detection.
- `train_model.py`: Training script for the custom CNN model.
- `model/model.h5`: Trained model file used by the app.
- `model/label_classes.npy`: Label mapping for model predictions.
- `utils/`: Dataset loading, feature extraction, and alert helpers.

## Prerequisites

- Python 3.10 or 3.11 recommended
- `pip` installed
- Microphone access (only for Live Detection tab)

## Step-by-step setup and run

### 1. Open project folder

```bash
cd Hazard-system-
```

### 2. Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## Optional: retrain the model

Use this only if you want to regenerate `model/model.h5`.

### 1. Download UrbanSound8K

- Download from: https://urbansounddataset.weebly.com/urbansound8k.html
- Extract and place it as:

```text
Hazard-system-/
    dataset/
        UrbanSound8K/
            audio/fold1 ... fold10
            metadata/UrbanSound8K.csv
```

### 2. Train

```bash
python train_model.py
```

Training outputs:

- `model/model.h5`
- `model/training_history.png`

## Common issues

- `ModuleNotFoundError`: ensure virtual environment is active and dependencies are installed.
- Microphone errors in live tab: install/check audio drivers and allow mic permission.
- If `sounddevice` install fails on Windows, install Microsoft C++ Build Tools and retry.

## Main dependencies

- `tensorflow`, `tensorflow-hub`
- `streamlit`
- `librosa`, `soundfile`, `scipy`
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`, `tqdm`, `sounddevice`
