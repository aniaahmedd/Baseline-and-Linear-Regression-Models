"""
Baseline and Linear Regression Models
======================================
Dataset: Diabetes disease-progression dataset (sklearn built-in, 442 patients)
Target: Quantitative measure of disease progression one year after baseline

Tasks covered:
1. Baseline prediction (mean predictor)
2. Simple Linear Regression (single best feature)
3. Multiple Linear Regression (all features)
4. Polynomial Regression (degree 2, on top features)
5. Predictions on test set
6. Evaluation: MAE, MSE, RMSE, R2, Adjusted R2, MAPE
7. Model comparison
8. Residual analysis + regression assumption checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
data = load_diabetes(as_frame=True)
df = data.frame.copy()
print("Dataset shape:", df.shape)
print(df.head())

FEATURES = [c for c in df.columns if c != "target"]
X = df[FEATURES]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# Helper: evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, n_features, model_name):
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    # Adjusted R2 (guard against n - p - 1 <= 0)
    denom = n - n_features - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / denom if denom > 0 else np.nan
    # MAPE (guard against zeros in y_true)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
        "MAPE (%)": mape,
    }

results = []

# ---------------------------------------------------------------------------
# 1. Baseline model: predict the mean of the training target for every case
# ---------------------------------------------------------------------------
baseline_pred_value = y_train.mean()
baseline_preds = np.full(shape=y_test.shape, fill_value=baseline_pred_value)

results.append(evaluate(y_test, baseline_preds, n_features=0, model_name="Baseline (Mean)"))
print(f"\nBaseline prediction (mean of y_train): {baseline_pred_value:.3f}")

# ---------------------------------------------------------------------------
# 2. Simple Linear Regression — pick the single feature most correlated with target
# ---------------------------------------------------------------------------
correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
best_feature = correlations.index[0]
print(f"\nMost correlated feature with target: {best_feature} (|r| = {correlations.iloc[0]:.3f})")

X_train_simple = X_train[[best_feature]]
X_test_simple = X_test[[best_feature]]

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)
simple_preds = simple_model.predict(X_test_simple)

results.append(evaluate(y_test, simple_preds, n_features=1, model_name=f"Simple Linear ({best_feature})"))
print(f"Simple LR coefficient: {simple_model.coef_[0]:.3f}, intercept: {simple_model.intercept_:.3f}")

# ---------------------------------------------------------------------------
# 3. Multiple Linear Regression — all features
# ---------------------------------------------------------------------------
multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
multi_preds = multi_model.predict(X_test)

results.append(evaluate(y_test, multi_preds, n_features=X_train.shape[1], model_name="Multiple Linear (all features)"))

coef_table = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": multi_model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nMultiple Linear Regression coefficients:")
print(coef_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Polynomial Regression (degree 2) — on top 3 correlated features
#    (keeps feature count reasonable and avoids severe overfitting)
# ---------------------------------------------------------------------------
top_features = correlations.index[:3].tolist()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[top_features])
X_test_poly = poly.transform(X_test[top_features])

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
poly_preds = poly_model.predict(X_test_poly)

results.append(evaluate(y_test, poly_preds, n_features=X_train_poly.shape[1],
                         model_name=f"Polynomial deg=2 ({', '.join(top_features)})"))

# ---------------------------------------------------------------------------
# 5 & 6. Comparison table
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("\n=== Model Performance Comparison ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

# ---------------------------------------------------------------------------
# 7. Residual analysis (on Multiple Linear Regression — the main model)
# ---------------------------------------------------------------------------
residuals = y_test - multi_preds

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(multi_preds, residuals, alpha=0.6, edgecolor="k")
axes[0, 0].axhline(0, color="red", linestyle="--")
axes[0, 0].set_xlabel("Fitted values")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].set_title("Residuals vs Fitted (linearity/homoscedasticity check)")

# Histogram of residuals
axes[0, 1].hist(residuals, bins=20, edgecolor="k", alpha=0.7)
axes[0, 1].set_xlabel("Residual")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Distribution of Residuals (normality check)")

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot (normality check)")

# Residuals vs each predicted order (independence proxy)
axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.6, edgecolor="k")
axes[1, 1].axhline(0, color="red", linestyle="--")
axes[1, 1].set_xlabel("Observation order")
axes[1, 1].set_ylabel("Residuals")
axes[1, 1].set_title("Residuals vs Order (independence check)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_analysis.png"), dpi=150)
plt.close()

# Shapiro-Wilk normality test on residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)

# Simple heteroscedasticity signal: correlation between |residuals| and fitted values
het_corr = np.corrcoef(np.abs(residuals), multi_preds)[0, 1]

print("\n=== Residual Analysis ===")
print(f"Mean residual: {residuals.mean():.4f} (should be close to 0)")
print(f"Shapiro-Wilk normality test: stat={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
print(f"  -> {'Residuals approx. normal (fail to reject H0)' if shapiro_p > 0.05 else 'Residuals deviate from normality (reject H0)'}")
print(f"Correlation(|residuals|, fitted values): {het_corr:.4f}")
print(f"  -> {'Low correlation, homoscedasticity plausible' if abs(het_corr) < 0.3 else 'Possible heteroscedasticity (variance changes with fitted value)'}")

print("\nCharts saved to:", os.path.join(OUTPUT_DIR, "residual_analysis.png"))
print("Comparison table saved to:", os.path.join(OUTPUT_DIR, "model_comparison.csv"))
