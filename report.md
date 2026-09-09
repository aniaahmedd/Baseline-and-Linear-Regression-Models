# Baseline and Linear Regression Models — Report

**Dataset:** Diabetes disease-progression dataset (442 patients, 10 baseline features, sklearn built-in). Target = quantitative disease progression score one year after baseline.

## 1. Baseline Model
Predicts the training mean (153.74) for every test case, as a reference point. R² ≈ -0.01 (essentially no predictive power), as expected for a constant predictor.

## 2. Simple Linear Regression
Feature used: **bmi** (highest correlation with target, |r| = 0.605).
R² = 0.233 — a large jump over baseline, confirming BMI alone carries real signal.

## 3. Multiple Linear Regression
Uses all 10 features. R² = 0.453, Adjusted R² = 0.382 — clear improvement over the single-feature model. Largest-magnitude coefficients: `s1`, `s5`, `bmi`, `s2` (blood serum markers and BMI dominate).

## 4. Polynomial Regression (degree 2)
Applied to the top 3 correlated features (bmi, s5, bp) to capture nonlinearity/interactions. R² = 0.467, Adjusted R² = 0.406 — best of the four models, though the gain over multiple linear regression is modest, suggesting the relationship is close to linear with mild curvature/interaction effects.

## 5. Model Comparison

| Model | MAE | MSE | RMSE | R² | Adj. R² | MAPE (%) |
|---|---|---|---|---|---|---|
| Baseline (Mean) | 64.01 | 5361.5 | 73.22 | -0.012 | -0.012 | 62.8 |
| Simple Linear (bmi) | 52.26 | 4061.8 | 63.73 | 0.233 | 0.225 | 45.9 |
| Multiple Linear (all features) | 42.79 | 2900.2 | 53.85 | 0.453 | 0.382 | 37.5 |
| Polynomial deg=2 (bmi, s5, bp) | 44.03 | 2824.9 | 53.15 | 0.467 | 0.406 | 38.8 |

Every model beats the baseline. Multiple linear regression gives the biggest single jump in performance (adding more predictors matters more than adding nonlinearity here). Polynomial regression edges out multiple linear regression slightly on R²/RMSE but not on MAE/MAPE, and its Adjusted R² gain is small relative to the added complexity (10 features → 9 polynomial terms from only 3 base features) — a sign of mild diminishing returns / early overfitting risk on this small dataset (n=442).

## 6. Residual Analysis (Multiple Linear Regression)
- **Mean residual ≈ 3.91**, close to 0 — no major systematic bias.
- **Residuals vs Fitted:** scattered fairly randomly around 0 with no strong curved pattern — mild support for linearity, though spread increases somewhat at higher fitted values.
- **Normality:** Shapiro-Wilk test p = 0.907 (fail to reject normality) and the Q-Q plot points fall close to the reference line — residuals are approximately normally distributed.
- **Homoscedasticity:** correlation between |residuals| and fitted values = 0.275 (moderate-low) — some mild heteroscedasticity is possible but not severe.
- **Independence:** residuals vs. observation order show no obvious trend/pattern, consistent with independent errors (data has no inherent time ordering here).

## 7. Possible Assumption Violations
- **Linearity:** Partially violated — the polynomial model's improvement, though small, suggests some nonlinear/interaction structure the linear model doesn't fully capture.
- **Homoscedasticity:** Borderline — variance of residuals appears to grow slightly with fitted values, worth monitoring if this model were extended.
- **Multicollinearity:** Not tested numerically here (e.g., VIF), but several serum markers (s1–s6) are likely correlated with each other, which can inflate coefficient variance in the multiple regression (note the large opposite-signed coefficients for `s1` and `s2`, a classic multicollinearity symptom).
- **Sample size:** With only 442 samples split into an ~88-row test set, metric estimates (especially R²) carry meaningful uncertainty; results should be treated as indicative rather than final.

## Files
- `regression_project.py` — full runnable pipeline
- `model_comparison.csv` — metrics table
- `residual_analysis.png` — 4-panel diagnostic plot
