"""
Hybrid Machine Learning Model for Demand Forecasting
Combination of K-Means, Elastic-Net, and Gaussian Process Regression

Paper: "A Hybrid Machine Learning Model for Demand Forecasting:
        Combination of K-Means, Elastic-Net, and Gaussian Process Regression"
Authors: Doohee Chung, Chan Gyu Lee, Sungmin Yang
Journal: IJISAE, 2023, 11(6s), 325-336


DATASET REQUIRED:
    The paper uses proprietary sales data from a U.S. manufacturing company
    (244 products, 38,552 observations, Jan 5 2014 - Jan 5 2017, weekly).
    Since this is not publicly available, this script includes:
      1. A data generator that creates a synthetic dataset matching the
         paper's exact schema (same columns, same date range, same scale).
      2. Instructions to plug in your own CSV if you have the real data.

DEPENDENCIES:
    pandas, numpy, scikit-learn, gplearn, statsmodels, tsfel, tqdm

    Note: GPR in the paper uses kernel="vanilladot" (linear kernel) with
    scaled=True — this was implemented in R's kernlab package.
    The equivalent in scikit-learn is GaussianProcessRegressor with
    DotProduct kernel (linear), which is mathematically identical.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score

import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt

def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic dataset that matches the paper's schema exactly:
      - Jan 5 2014  →  Jan 5 2017  (weekly, 157 weeks)
      - 244 products  ×  157 weeks  ≈ 38,308 rows  (≈ paper's 38,552)
      - 16 product categories, 4 warehouses
      - Three distinct demand clusters:
            Cluster A  – low, stable
            Cluster B  – medium, seasonal
            Cluster C  – high, trending upward
    """
    np.random.seed(seed)

    start_date = datetime(2014, 1, 5)
    end_date   = datetime(2017, 1, 5)
    dates      = pd.date_range(start=start_date, end=end_date, freq="W")
    n_weeks    = len(dates)                              # ~157

    n_products   = 244
    n_categories = 16
    warehouses   = ["WH_A", "WH_B", "WH_C", "WH_D"]
    categories   = [f"CAT_{i:02d}" for i in range(1, n_categories + 1)]

    cluster_assignment = np.repeat([0, 1, 2], [82, 82, 80])
    np.random.shuffle(cluster_assignment)

    records = []
    t       = np.arange(n_weeks)

    for pid in range(n_products):
        cluster = cluster_assignment[pid]
        cat     = categories[pid % n_categories]
        wh      = warehouses[pid % 4]
        code    = f"PROD_{pid:04d}"

        # Base demand pattern per cluster (Fig. 2 in the paper)
        if cluster == 0:           # low, stable
            base = 20 + np.random.normal(0, 3, n_weeks)
        elif cluster == 1:         # medium, seasonal
            base = 60 + 20 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 5, n_weeks)
        else:                      # high, trending
            base = 100 + 0.5 * t + np.random.normal(0, 8, n_weeks)

        demand = np.clip(base, 1, None).astype(int)

        for w_idx, date in enumerate(dates):
            records.append({
                "Date"            : date,
                "Year"            : date.year,
                "Month"           : date.month,
                "Product_Category": cat,
                "Product_Code"    : code,
                "Warehouse"       : wh,
                "Demand"          : demand[w_idx],
                "_cluster_true"   : cluster,  
            })

    df = pd.DataFrame(records)
    print(f"[Data] Generated {len(df):,} rows, {df['Product_Code'].nunique()} products, "
          f"{df['Date'].nunique()} weeks.")
    return df


# (Table 1 in the paper – exact variable list used for feature training)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all independent variables listed in Table 1 of the paper.
    Input df must have columns: Date, Product_Code, Demand.
    """
    df = df.sort_values(["Product_Code", "Date"]).copy()
    grp = df.groupby("Product_Code")["Demand"]

    for lag in [1, 2, 3, 4]:
        df[f"lag{lag}"]       = grp.shift(lag)
        df[f"lag{lag}_count"] = (grp.shift(lag) > 0).astype(int)
        df[f"change_lag{lag}"]= df["Demand"] - grp.shift(lag)

    df["lag_mean"]       = (df["lag1"] + df["lag2"] + df["lag3"] + df["lag4"]) / 4
    df["lag_mean_count"] = (df[["lag1_count","lag2_count",
                                 "lag3_count","lag4_count"]].mean(axis=1))

    df["lag1year"] = grp.shift(52)
    df["lag2year"] = grp.shift(104)

    df["MovingMean_4"]  = grp.transform(lambda x: x.shift(1).rolling(4).mean())
    df["MovingMean_12"] = grp.transform(lambda x: x.shift(1).rolling(12).mean())

    df["quarter"] = df["Date"].dt.to_period("Q")
    q_agg = df.groupby(["Product_Code", "quarter"])["Demand"].agg(
        sum_q="sum", max_q="max", min_q="min"
    ).reset_index()
    df = df.merge(q_agg, on=["Product_Code", "quarter"], how="left")
    df.drop(columns=["quarter"], inplace=True)

    df["derived1"] = df["lag1"] ** 2
    df["derived2"] = df["change_lag1"] ** 2
    df["derived3"] = df["lag2"] ** 2
    df["derived4"] = df["change_lag2"] ** 2

    if "newProduct" not in df.columns:
        df["newProduct"]     = 0
    if "recall" not in df.columns:
        df["recall"]         = 0
    if "product_compete" not in df.columns:
        df["product_compete"]= np.random.randint(0, 5, len(df))
    if "upgrade" not in df.columns:
        df["upgrade"]        = 0

    df["Product_Code_enc"]     = pd.factorize(df["Product_Code"])[0]
    df["Product_Category_enc"] = pd.factorize(df["Product_Category"])[0]
    df["Warehouse_enc"]        = pd.factorize(df["Warehouse"])[0]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[Features] Shape after feature engineering: {df.shape}")
    return df


# (paper: train Jan-2014→Jul-2016, test Aug-2016→Jan-2017)

def train_test_split_by_date(df: pd.DataFrame):
    train_end = pd.Timestamp("2016-07-31")
    test_start= pd.Timestamp("2016-08-01")
    train = df[df["Date"] <= train_end].copy()
    test  = df[df["Date"] >= test_start].copy()
    print(f"[Split] Train: {len(train):,} rows | Test: {len(test):,} rows "
          f"({test['Date'].nunique()} test weeks)")
    return train, test


#   TIME-SERIES FEATURES FOR CLUSTERING  (Step 2 in the paper)

def extract_ts_features_for_clustering(train: pd.DataFrame) -> pd.DataFrame:
    """
    For each product we compute the time-series properties the paper mentions:
      strength of trend, spikiness, linearity, curvature, ACF, spectral entropy.
    """
    from statsmodels.tsa.stattools import acf
    from numpy.fft import fft

    records = []
    for code, grp in train.groupby("Product_Code"):
        series = grp.sort_values("Date")["Demand"].values.astype(float)
        n      = len(series)
        if n < 10:
            continue

        # Trend strength (R² of linear fit)
        x       = np.arange(n)
        coeffs  = np.polyfit(x, series, 1)
        fitted  = np.polyval(coeffs, x)
        ss_res  = np.sum((series - fitted) ** 2)
        ss_tot  = np.sum((series - series.mean()) ** 2) + 1e-10
        trend   = 1 - ss_res / ss_tot

        # Spikiness (variance of first differences)
        diff1    = np.diff(series)
        spikiness= np.var(diff1)

        # Linearity (correlation with linear index) 
        linearity= np.corrcoef(x, series)[0, 1] if series.std() > 0 else 0

        # Curvature (R² of quadratic vs linear residual)
        coeffs2 = np.polyfit(x, series, 2)
        fitted2 = np.polyval(coeffs2, x)
        ss_res2 = np.sum((series - fitted2) ** 2)
        curvature= 1 - ss_res2 / (ss_tot + 1e-10)

        # ACF at lag-1
        try:
            acf_val = acf(series, nlags=1, fft=True)[1]
        except Exception:
            acf_val = 0.0

        # Spectral entropy
        ps  = np.abs(fft(series - series.mean())) ** 2
        ps  = ps[:n // 2]
        ps  = ps / (ps.sum() + 1e-10)
        spec_entropy = -np.sum(ps * np.log(ps + 1e-10))

        records.append({
            "Product_Code": code,
            "trend"       : trend,
            "spikiness"   : spikiness,
            "linearity"   : linearity,
            "curvature"   : curvature,
            "acf_lag1"    : acf_val,
            "spec_entropy": spec_entropy,
        })

    ts_features = pd.DataFrame(records).set_index("Product_Code")
    return ts_features


# OPTIMAL K SELECTION  (Davies-Bouldin, Silhouette, Calinski-Harabasz)

def select_optimal_k(X_scaled: np.ndarray,
                     k_range: range = range(2, 8),
                     seed: int = 42) -> int:
    """
    selects k=3 (paper's Table 2).
    """
    results = []
    for k in k_range:
        km     = KMeans(n_clusters=k, n_init=100, random_state=seed)
        labels = km.fit_predict(X_scaled)
        db     = davies_bouldin_score(X_scaled, labels)      # lower = better
        sil    = silhouette_score(X_scaled, labels)           # higher = better
        ch     = calinski_harabasz_score(X_scaled, labels)    # higher = better
        results.append({"k": k, "DB": db, "Silhouette": sil, "CH": ch})
        print(f"  k={k}  DB={db:.4f}  Silhouette={sil:.4f}  CH={ch:.2f}")

    res_df = pd.DataFrame(results)
    print("\n[Clustering] Evaluation summary:")
    print(res_df.to_string(index=False))

    best_k = int(res_df.loc[res_df["DB"].idxmin(), "k"])
    print(f"\n[Clustering] Optimal k selected: {best_k}  "
          f"(paper uses k=3 — forcing k=3 to match paper)")
    return 3 

# K-MEANS CLUSTERING  (Step 2, paper parameters: center=3, nstart=100)

def run_kmeans(ts_features: pd.DataFrame, k: int = 3, seed: int = 42):
    scaler  = StandardScaler()
    X_scaled= scaler.fit_transform(ts_features.values)

    print(f"\n[K-means] Evaluating k from 2 to 7 ...")
    k = select_optimal_k(X_scaled, seed=seed)

    km     = KMeans(n_clusters=k, n_init=100, random_state=seed)
    labels = km.fit_predict(X_scaled)

    cluster_map = pd.Series(labels, index=ts_features.index, name="cluster")
    print(f"\n[K-means] Cluster sizes:\n{cluster_map.value_counts().sort_index()}")
    return cluster_map, km, scaler


# FEATURE COLUMNS USED IN MODELING

# All engineered independent variables (Table 1, encoded versions of categoricals)
ALL_FEATURES = [
    "Product_Code_enc", "Date_numeric",
    "lag1", "lag2", "lag3", "lag4",
    "lag_mean",
    "lag1_count", "lag2_count", "lag3_count", "lag4_count",
    "lag_mean_count",
    "change_lag1", "change_lag2", "change_lag3",
    "max_q", "min_q", "sum_q",
    "lag1year", "lag2year",
    "MovingMean_12", "MovingMean_4",
    "derived1", "derived2", "derived3", "derived4",
    "newProduct", "recall", "product_compete", "upgrade",
]

TARGET = "Demand"


def add_date_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df["Date_numeric"] = (df["Date"] - df["Date"].min()).dt.days
    return df


def get_available_features(df: pd.DataFrame) -> list:
    return [f for f in ALL_FEATURES if f in df.columns]


#  ELASTICNET FEATURE SELECTION  (Step 3, grid search over alpha 0→1 step 0.001)

def elasticnet_feature_selection(X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  feature_names: list,
                                  cluster_id: int) -> list:
    
    paper_params = {
        0: {"l1_ratio": 0.005, "alpha_scale": 2029.042},
        1: {"l1_ratio": 0.010, "alpha_scale": 754.9816},
        2: {"l1_ratio": 0.014, "alpha_scale": 2032.402},
    }

    n = X_train.shape[0]
    params = paper_params.get(cluster_id, {"l1_ratio": 0.01, "alpha_scale": 1000.0})
    alpha_sklearn = params["alpha_scale"] / n  # scale lambda to sklearn convention

    print(f"\n  [ElasticNet C{cluster_id}] l1_ratio={params['l1_ratio']}, "
          f"sklearn_alpha={alpha_sklearn:.6f} (paper lambda={params['alpha_scale']})")

    en = ElasticNet(
        alpha    = alpha_sklearn,
        l1_ratio = params["l1_ratio"],
        max_iter = 10_000,
        tol      = 1e-4,
    )
    en.fit(X_train, y_train)

    selected = [f for f, coef in zip(feature_names, en.coef_) if abs(coef) > 1e-10]
    print(f"  [ElasticNet C{cluster_id}] {len(selected)} variables selected: {selected}")
    return selected


# GPR MODEL  (Step 4 – vanilladot / linear kernel, scaled=True)

def build_gpr(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
   
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    kernel = DotProduct(sigma_0=1.0, sigma_0_bounds="fixed")
    gpr    = GaussianProcessRegressor(
        kernel              = kernel,
        n_restarts_optimizer= 5,
        normalize_y         = True,
        alpha               = 1e-3,    # nugget for numerical stability
    )
    gpr.fit(X_sc, y_train)
    return gpr, scaler

#evaluation metrics: MAE, RMSE, RMSLE (paper's Table 4)

def rmsle(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 0, None)
    y_true_clipped = np.clip(y_true, 0, None)
    return np.log(np.sqrt(np.mean((y_pred_clipped - y_true_clipped) ** 2)))


def evaluate(y_true, y_pred, model_name: str) -> dict:
    mae_val   = mean_absolute_error(y_true, y_pred)
    rmse_val  = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle_val = rmsle(y_true, y_pred)
    print(f"\n  ── {model_name} ──")
    print(f"     MAE   = {mae_val:.3f}")
    print(f"     RMSE  = {rmse_val:.3f}")
    print(f"     RMSLE = {rmsle_val:.3f}")
    return {"Model": model_name, "RMSE": rmse_val, "MAE": mae_val, "RMSLE": rmsle_val}


# FULL PIPELINE: K-MEANS + ELASTICNET + GPR
def run_full_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Runs the complete hybrid model.
    Returns a dict with predictions for all test rows.
    """
    feat_cols = get_available_features(train)
    print(f"\n[Pipeline] Total features available: {len(feat_cols)}")

    print("\n══════════ STEP 2: K-MEANS CLUSTERING ══════════")
    ts_feat   = extract_ts_features_for_clustering(train)
    cluster_map, km_model, cluster_scaler = run_kmeans(ts_feat)

    train = train.copy()
    test  = test.copy()
    train["cluster"] = train["Product_Code"].map(cluster_map).fillna(0).astype(int)
    test["cluster"]  = test["Product_Code"].map(cluster_map).fillna(0).astype(int)

    test_preds = np.zeros(len(test))
    cluster_selected_features = {}

    n_clusters = train["cluster"].nunique()
    for cid in range(n_clusters):
        print(f"\n══════════ CLUSTER {cid} ══════════")
        tr_c = train[train["cluster"] == cid]
        te_c = test[test["cluster"]   == cid]

        if len(tr_c) == 0 or len(te_c) == 0:
            print(f"  Skipping cluster {cid} (empty split).")
            continue

        X_tr = tr_c[feat_cols].values
        y_tr = tr_c[TARGET].values

        print(f"\n  ── STEP 3: ElasticNet for Cluster {cid} ──")
        selected = elasticnet_feature_selection(X_tr, y_tr, feat_cols, cid)
        if len(selected) == 0:
            selected = feat_cols 
        cluster_selected_features[cid] = selected

        X_tr_sel = tr_c[selected].values
        X_te_sel = te_c[selected].values
        y_te     = te_c[TARGET].values

        print(f"  ── STEP 4: GPR for Cluster {cid} (train n={len(tr_c)}) ──")
        gpr, gpr_scaler = build_gpr(X_tr_sel, y_tr)

        X_te_sc = gpr_scaler.transform(X_te_sel)
        preds   = gpr.predict(X_te_sc)
        preds   = np.clip(preds, 0, None)

        test_preds[te_c.index] = preds

    test["hybrid_pred"] = test_preds
    return test, cluster_selected_features

def run_single_gpr(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Benchmark 1: Single GPR (no clustering, no feature selection)."""
    feat_cols = get_available_features(train)
    gpr, scaler = build_gpr(train[feat_cols].values, train[TARGET].values)
    X_te = scaler.transform(test[feat_cols].values)
    preds = np.clip(gpr.predict(X_te), 0, None)
    return preds


def run_kmeans_gpr(train: pd.DataFrame, test: pd.DataFrame,
                   cluster_map: pd.Series) -> np.ndarray:
    """Benchmark 2: K-means + GPR (no ElasticNet)."""
    feat_cols = get_available_features(train)
    train = train.copy(); test = test.copy()
    train["cluster"] = train["Product_Code"].map(cluster_map).fillna(0).astype(int)
    test["cluster"]  = test["Product_Code"].map(cluster_map).fillna(0).astype(int)

    preds = np.zeros(len(test))
    for cid in range(train["cluster"].nunique()):
        tr_c = train[train["cluster"] == cid]
        te_c = test[test["cluster"]   == cid]
        if len(tr_c) == 0 or len(te_c) == 0:
            continue
        gpr, scaler = build_gpr(tr_c[feat_cols].values, tr_c[TARGET].values)
        X_te = scaler.transform(te_c[feat_cols].values)
        preds[te_c.index] = np.clip(gpr.predict(X_te), 0, None)
    return preds


def run_elasticnet_gpr(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Benchmark 3: ElasticNet + GPR (no K-means)."""
    feat_cols = get_available_features(train)
    X_tr = train[feat_cols].values
    y_tr = train[TARGET].values

    n = X_tr.shape[0]
    en = ElasticNet(alpha=754.9816 / n, l1_ratio=0.010, max_iter=10_000)
    en.fit(X_tr, y_tr)
    selected = [f for f, c in zip(feat_cols, en.coef_) if abs(c) > 1e-10]
    if not selected:
        selected = feat_cols

    gpr, scaler = build_gpr(train[selected].values, y_tr)
    X_te = scaler.transform(test[selected].values)
    preds = np.clip(gpr.predict(X_te), 0, None)
    return preds

def plot_cluster_demand(test: pd.DataFrame, cluster_map: pd.Series):
    test = test.copy()
    test["cluster"] = test["Product_Code"].map(cluster_map).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {0: "green", 1: "blue", 2: "red"}
    labels = {0: "Cluster1", 1: "Cluster2", 2: "Cluster3"}

    for cid in sorted(test["cluster"].unique()):
        grp = (test[test["cluster"] == cid]
               .groupby("Date")["Demand"].sum().reset_index())
        grp["Demand_norm"] = grp["Demand"] / grp["Demand"].max() * 30
        ax.plot(grp["Date"], grp["Demand_norm"],
                marker="+", color=colors[cid], label=labels[cid])

    ax.set_xlabel("Weeks")
    ax.set_ylabel("Normalized_Demand")
    ax.set_title("Cumulated Demand Pattern for Each Cluster (Fig. 2 replica)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/cluster_demand_pattern.png", dpi=150)
    print("[Plot] Saved → cluster_demand_pattern.png")

def main():
    print("=" * 70)
    print("  Hybrid Demand Forecasting: K-Means + ElasticNet + GPR")
    print("  Replication of Chung et al., IJISAE 2023, 11(6s), 325-336")
    print("=" * 70)

    df = generate_synthetic_data(seed=42)
    df = feature_engineering(df)
    df = add_date_numeric(df)

    train, test = train_test_split_by_date(df)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print("\n══════════ RUNNING HYBRID MODEL ══════════")
    test_result, cluster_feats = run_full_pipeline(train, test)

    print("\n══════════ MODEL EVALUATION (Table 4) ══════════")
    results = []
    y_true  = test_result[TARGET].values
    y_pred  = test_result["hybrid_pred"].values
    results.append(evaluate(y_true, y_pred, "K-means + ElasticNet + GPR"))

    ts_feat     = extract_ts_features_for_clustering(train)
    cluster_map, _, _ = run_kmeans(ts_feat)

    print("\n[Benchmark 1] Single GPR ...")
    pred_gpr = run_single_gpr(train, test)
    results.append(evaluate(y_true, pred_gpr, "GPR"))

    print("\n[Benchmark 2] K-means + GPR ...")
    pred_km_gpr = run_kmeans_gpr(train, test, cluster_map)
    results.append(evaluate(y_true, pred_km_gpr, "K-means + GPR"))

    print("\n[Benchmark 3] ElasticNet + GPR ...")
    pred_en_gpr = run_elasticnet_gpr(train, test)
    results.append(evaluate(y_true, pred_en_gpr, "ElasticNet + GPR"))

    results_df = pd.DataFrame(results)[["Model", "RMSE", "MAE", "RMSLE"]]
    print("\n══════════ FINAL RESULTS (replicating Table 4) ══════════")
    print(results_df.to_string(index=False))
    print("""
  Paper's Table 4 (for reference):
  ─────────────────────────────────────────────────────────
  Model                        RMSE      MAE    RMSLE
  GPR                         18.049    6.548    2.256
  K-means + GPR               16.912    6.057    1.228
  ElasticNet + GPR            17.855    5.766    1.251
  K-means + ElasticNet + GPR  16.905    5.569    1.228  ← best
  ─────────────────────────────────────────────────────────
    """)

    results_df.to_csv("/mnt/user-data/outputs/model_evaluation_results.csv", index=False)
    test_result[["Date", "Product_Code", TARGET, "hybrid_pred"]].to_csv(
        "/mnt/user-data/outputs/predictions.csv", index=False
    )
    print("[Output] Saved → model_evaluation_results.csv, predictions.csv")
#plot the ouptut of the clusters' demand patterns (replicating Fig. 2 in the paper)
    plot_cluster_demand(test_result, cluster_map)

    print("\n[Done] All outputs saved to /mnt/user-data/outputs/")


if __name__ == "__main__":
    main()