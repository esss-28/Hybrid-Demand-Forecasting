# 📊 Hybrid Demand Forecasting Model

**K-Means + ElasticNet + Gaussian Process Regression**

---

## 🧠 Overview

This project implements a **hybrid machine learning pipeline for demand forecasting**, inspired by the paper:

> *“A Hybrid Machine Learning Model for Demand Forecasting: Combination of K-Means, Elastic-Net, and Gaussian Process Regression”*  
> (Chung et al., IJISAE, 2023)

The system combines:

- **Unsupervised learning (K-Means)** → to segment products based on demand behavior  
- **Regularized regression (ElasticNet)** → to select the most relevant features per cluster  
- **Probabilistic modeling (Gaussian Process Regression)** → to perform final demand prediction  

This modular design improves forecasting accuracy by **capturing heterogeneity across product demand patterns**.

---

## ⚙️ Key Features

- 📦 **Synthetic Data Generator** (mirrors proprietary dataset structure)
- 🧮 **Advanced Feature Engineering** (lags, rolling stats, seasonality, derived features)
- 🔍 **Time-Series Clustering** using statistical descriptors
- 🎯 **Cluster-Specific Feature Selection**
- 📈 **Gaussian Process Regression (GPR)** with linear kernel
- 🧪 **Benchmark Models** for comparison:
  - GPR only
  - K-Means + GPR
  - ElasticNet + GPR
- 📊 **Evaluation Metrics**: RMSE, MAE, RMSLE
- 📉 **Visualization of demand patterns**

---

## 🗂️ Dataset

### 🔹 Synthetic Data (Built-in)

Since the original dataset is proprietary, this implementation includes a generator that replicates:

- **244 products**
- **~38,000 observations**
- **Weekly data (2014–2017)**
- **16 product categories, 4 warehouses**
- 3 demand patterns:
  - Low & stable
  - Seasonal
  - Trending upward

---

### 🔹 Using Your Own Data

To use real data, ensure your dataset contains:

```text
Date, Product_Code, Product_Category, Warehouse, Demand
```

Then replace:

```python
df = generate_synthetic_data()
```

with:

```python
df = pd.read_csv("your_dataset.csv")
```

---

## 🏗️ Pipeline Architecture

### Step 1 — Feature Engineering

Creates variables such as:

- Lag features (`lag1`, `lag2`, ...)
- Moving averages
- Yearly lags
- Demand change features
- Encoded categorical variables

---

### Step 2 — Time-Series Clustering

Each product is transformed into statistical descriptors:

- Trend strength (R²)
- Spikiness (variance of differences)
- Linearity & curvature
- Autocorrelation
- Spectral entropy

Then:

- Standardization
- Optimal **K selection** (Davies-Bouldin, Silhouette, CH)
- K-Means clustering (k = 3 as per paper)

---

### Step 3 — Feature Selection (ElasticNet)

For each cluster:

- ElasticNet selects the most predictive variables
- Uses paper-specific hyperparameters
- Eliminates irrelevant or redundant features

---

### Step 4 — Prediction (GPR)

- Gaussian Process Regression with **DotProduct kernel (linear)**
- Equivalent to “vanilladot” kernel used in the paper
- Cluster-specific models trained independently

---

## 📊 Model Evaluation

Metrics used:

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **RMSLE** – Log-scaled error  

The script reproduces **Table 4 from the paper** and compares:

| Model            | Description                      |
|------------------|----------------------------------|
| GPR              | Baseline                         |
| K-Means + GPR    | Adds clustering                  |
| ElasticNet + GPR | Adds feature selection           |
| **Hybrid Model** | Full pipeline (best performance) |

---

## 📈 Outputs

All results are saved to:

```
/mnt/user-data/outputs/
```

### Generated Files:

- `model_evaluation_results.csv` → metrics comparison  
- `predictions.csv` → actual vs predicted demand  
- `cluster_demand_pattern.png` → visualization of clusters  

---

## 📦 Dependencies

Install required libraries:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib tqdm
```

---

## 🚀 How to Run

```bash
python hybrid_demand_forecasting.py
```

The script will:

1. Generate data  
2. Engineer features  
3. Split train/test  
4. Run hybrid pipeline  
5. Evaluate benchmarks  
6. Save outputs  

---

## 🧩 Code Structure

| Function                               | Purpose                 |
|----------------------------------------|-------------------------|
| `generate_synthetic_data()`            | Creates dataset         |
| `feature_engineering()`                | Builds features         |
| `extract_ts_features_for_clustering()` | Time-series descriptors |
| `run_kmeans()`                         | Clustering              |
| `elasticnet_feature_selection()`       | Feature selection       |
| `build_gpr()`                          | Train GPR model         |
| `run_full_pipeline()`                  | Hybrid workflow         |
| `evaluate()`                           | Metrics                 |
| `main()`                               | Entry point             |

---

## 🧪 Reproducibility Notes

- Fixed random seed (`seed=42`)
- Matches paper:
  - Dataset structure
  - Feature definitions
  - Model architecture
- Slight deviations:
  - Synthetic data instead of proprietary
  - sklearn implementation of GPR

---

## 🔬 Research Insight

This hybrid architecture works because:

- **Clustering reduces distributional heterogeneity**
- **ElasticNet reduces dimensional noise**
- **GPR models uncertainty and nonlinearity**

Together, they form a **bias-variance optimized system**:

- Lower bias than simple models  
- Lower variance than raw GPR on full feature space  
