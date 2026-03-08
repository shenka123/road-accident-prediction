# Road Accident Risk Prediction

A machine learning project for the [Kaggle Playground Series S5E10](https://www.kaggle.com/competitions/playground-series-s5e10/overview) competition, which challenges participants to predict a continuous **accident risk score** for road segments based on environmental and road condition features.

**Author:** Benediktus Sashenka

## Problem Statement

Given a set of road and environmental attributes, predict the `accident_risk` score for each road segment. This is a **regression** task evaluated using **Root Mean Squared Error (RMSE)**.


## Dataset & Features

The dataset is provided by Kaggle (Playground Series S5E10). It includes `train.csv` and `test.csv` files with the following features:

| Type | Features |
|---|---|
| **Numerical** | `num_lanes`, `curvature`, `speed_limit`, `num_reported_accidents` |
| **Binary** | `road_signs_present`, `public_road`, `holiday`, `school_season` |
| **Categorical** | `road_type`, `lighting`, `weather`, `time_of_day` |
| **Target** | `accident_risk` (continuous score) |

> **Note:** The dataset files (`train.csv`, `test.csv`) in this repository only contains subset of the dataset. To obtain the full dataset, download from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s5e10/data).


## Project Structure

This repository content is as follows:
- `Road Accident Prediction Models (codes).ipynb`: main jupyter notebook   # Main Jupyter Notebook
- `Road Accident Prediction Models.pdf`: experiment methodology and results
- `pickle/`: saved pre-trained results and model checkpoints                    
- `submissions/`: Kaggle submission, i.e. model prediction of `test.csv`
---

## Methodology

The notebook is structured into three main parts:

### Part 1 — Data Preparation

- **Exploratory Data Analysis:** Distribution plots for all categorical, binary, and numerical features; box plots against the target variable.
- **Feature Engineering:**
  - `light_level` — composite score derived from `weather` and `lighting` to capture overall visibility.
  - `num_reported_accidents_sigmoid` — sigmoid-transformed version of accident history to compress outliers.
  - `speed_limit_high` — binary flag for speed limits ≥ 60.
- **Encoding:** One-hot encoding for categorical features; binary features cast to float.
- **Normalization:** Min-Max scaling applied to all numerical features.
- **PCA:** Standard scaling followed by Principal Component Analysis explored as an alternative feature representation (used with KNN models).

### Part 2 — Model Exploration

All models are evaluated using **5-fold cross-validation** with RMSE as the metric.

| Model | Hyperparameter Search Strategy |
|---|---|
| Linear Regression (+ Ridge) | Fixed `α = 0.5` |
| K-Nearest Neighbors | Sweep `k` from 2–20; uniform vs. distance weights; with/without PCA |
| Decision Tree | Sweep `max_depth` (4–16); grid search over `min_samples_split` & `min_samples_leaf` |
| Random Forest | Grid search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| LightGBM | Bayesian optimisation with **Optuna** (500 trials) |

### Part 3 — Final Submission

Two final models are trained on the full training set and used to generate Kaggle submissions:

- **Random Forest** — `n_estimators=1000`, `max_depth=11`, `min_samples_split=50`, `min_samples_leaf=2`
- **LightGBM** — best parameters found by Optuna hyperparameter search

---

## Models & Results

| Model | Cross-val RMSE |
|---|---|
| Linear Regression | 0.05916 |
| KNN | 0.05767 |
| Decision Tree | 0.05628 |
| Random Forest | 0.05609 |
| **LightGBM** | **0.05596** |

> Full results, plots, and analysis are available in **Road Accident Prediction Models.pdf**.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shenka123/road-accident-prediction.git
cd road-accident-prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn lightgbm optuna matplotlib seaborn jupyter
```

### 3. Download the dataset

Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s5e10/data) and place them in the root directory.

### 4. Run the notebook

```bash
jupyter notebook "Road Accident Prediction Models (codes).ipynb"
```

> Most training cells are commented out by default (to avoid long runtimes) and will load pre-saved results from the `pickle/` folder instead. Uncomment the training blocks if you want to retrain from scratch.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, models, cross-validation, PCA |
| `lightgbm` | Gradient boosting model |
| `optuna` | Bayesian hyperparameter optimisation |
| `matplotlib`, `seaborn` | Visualisations |
| `pickle` | Saving & loading model artefacts |
