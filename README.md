# Medical Insurance Charges — EDA & Linear Regression

Predicting individual medical insurance charges from demographic and lifestyle
data, using exploratory data analysis, data cleaning, feature engineering, and
a polynomial linear regression model.

## Dataset

[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(`insurance.csv`) — 1,338 records with `age`, `sex`, `bmi`, `children`,
`smoker`, `region`, and `charges` (the target).

## Approach

1. **EDA** — distributions, boxplots by category, correlation heatmap, and
   scatter plots to see which features actually relate to `charges`.
2. **Cleaning** — removed duplicate rows, confirmed no missing values,
   checked for outliers (kept them — they reflect real high-cost cases,
   mostly smokers, rather than data errors).
3. **Feature selection** — `smoker` is by far the strongest driver of cost;
   `age` and `bmi` also matter. `sex` and `region` showed no meaningful
   effect on charges and were excluded from the model.
4. **Modeling** — `RobustScaler` on `age`/`bmi` (robust to the outliers
   found in EDA), degree-2 polynomial features to capture non-linear /
   interaction effects, then `LinearRegression`.
5. **Evaluation** — compared against a naive baseline that always predicts
   the mean training charge.

## Results

| Metric | Baseline (mean) | Model |
|---|---|---|
| MAE | 9,861.80 | **2,814.39** |
| RMSE | 13,612.43 | **4,626.05** |
| R² | -0.84 | **88.35%** |

The model explains ~88% of the variance in charges and cuts RMSE by 66%
compared to just predicting the average.

## What I'd try next

- Explicit `smoker × bmi` interaction term, or a tree-based model
  (Random Forest / Gradient Boosting), which tends to capture that
  interaction more naturally than polynomial terms.
- Cross-validation instead of a single train/test split.
- Ridge/Lasso regularization on the polynomial features.

## Files

- `insurance_charges_prediction.ipynb` — full notebook (EDA → cleaning →
  modeling → evaluation), with outputs included so it renders on GitHub
  without needing to be re-run.
- `insurance.csv` — dataset.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook insurance_charges_prediction.ipynb
```
