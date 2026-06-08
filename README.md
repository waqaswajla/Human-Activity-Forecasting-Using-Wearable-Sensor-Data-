# Human Activity Forecasting Using Wearable Sensor Data

A machine learning project that forecasts the next human activity from accelerometer and gyroscope sequences recorded by a wrist-worn sensor. Three models are trained and compared: LSTM, GRU, and Random Forest.

---

## Problem Statement

Given a sliding window of 100 consecutive sensor readings, predict the activity the user will perform next. The five target classes are: LAYING, RUNNING, SITTING, STANDING, and WALKING.

---

## Dataset

The dataset is self-collected using a smartphone worn on the wrist during real-world activity sessions. It contains 75,263 timestamped sensor readings across six channels.

| Column | Description |
|--------|-------------|
| timestamp_ms | Unix timestamp in milliseconds |
| label | Activity class (5 classes) |
| ax, ay, az | Accelerometer readings (m/s²) |
| gx, gy, gz | Gyroscope readings (rad/s) |

The CSV file is not included in this repository due to file size. Place `merged_dataset.csv` in the project root before running the notebook.

---

## Project Structure

```
├── Activity_prediction.ipynb   # Main notebook
├── .gitignore
└── README.md
```

Generated after running the notebook:

```
├── lstm_model.keras
├── gru_model.keras
├── rf_model.pkl
├── scaler.pkl
└── label_encoder.pkl
```

---

## Notebook Overview

The notebook contains 37 numbered sections organized into five stages:

| Stage | Sections | Description |
|-------|----------|-------------|
| Exploratory Data Analysis | 1 - 11 | Load data, inspect structure, visualize distributions and correlations |
| Preprocessing | 12 - 18 | Sliding window sequencing, class balancing, train/test split, scaling, label encoding |
| Model Training | 19 - 23 | LSTM and GRU definition, training with early stopping, training history plots |
| Evaluation | 24 - 31 | Classification reports and confusion matrices for all three models |
| Inference and Saving | 32 - 37 | Save models, compare accuracy, run single and multi-sample predictions |

---

## Models

**LSTM**
- Architecture: LSTM (64 units) → Dropout (0.3) → Dense softmax (5 classes)
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Training: Up to 30 epochs with EarlyStopping (patience=5, restore best weights)

**GRU**
- Same architecture as the LSTM using a GRU layer
- Fewer parameters than LSTM, typically converges faster

**Random Forest**
- 100 estimators
- Sequences flattened to 1D feature vectors before training
- Validated with 5-fold cross-validation on the training set

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Stride = 10 | Reduces window overlap from 99% to 90%, lowering similarity between train and test samples |
| Scaler fit on training data only | Prevents test set statistics from leaking into the training pipeline |
| Balance after windowing | Ensures equal class representation at the sequence level, not just the raw sample level |
| EarlyStopping with restore_best_weights | Stops training when validation accuracy plateaus and keeps the best checkpoint |

---

## Setup

**Install dependencies**

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

**Run**

1. Place `merged_dataset.csv` in the project root.
2. Open `Activity_prediction.ipynb`.
3. Run all cells: `Kernel > Restart & Run All`.

---

## Results

| Model | Test Accuracy |
|-------|--------------|
| LSTM  | ~88% |
| GRU   | ~94% |
| Random Forest | ~99% |

---

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn
