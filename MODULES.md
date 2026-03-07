# Module Documentation

> How each notebook works, what it expects, what it produces, and how to customise it.
> Read this before modifying any notebook to understand the dependencies between them.

---

## Data Flow Overview

```
[Kaggle CSVs × 9]
       │
       ▼
01_ingestion_and_cleaning.ipynb
       │
       └── processed/master_orders.csv
                    │
       ┌────────────┼────────────────────┐
       ▼            ▼                    ▼
  02_eda       03_churn_model     04_demand_forecasting
  (no output)       │                    │
               churn_predictions    demand_forecasts
               .csv + .pkl          .csv
                    │
                    ▼
              05_segmentation
                    │
               customer_segments.csv
               churn_with_segments.csv
               kmeans_segmentation.pkl
               rfm_scaler.pkl
```

**Rule:** always run notebooks in order (01 → 02 → 03 → 04 → 05).
Notebooks 03, 04, and 05 all depend on `master_orders.csv` from notebook 01.
Notebook 05 additionally depends on `churn_predictions.csv` from notebook 03.

---

## Notebook 01 — Ingestion & Cleaning

**File:** `01_ingestion_and_cleaning.ipynb`

### What it does
Loads all 9 raw CSVs into pandas DataFrames, cleans nulls, parses timestamps,
translates Portuguese category names, aggregates multi-row tables (items, payments)
to order level, and joins everything into a single wide master DataFrame.

### Inputs
```
data/raw/olist_orders_dataset.csv
data/raw/olist_order_items_dataset.csv
data/raw/olist_order_payments_dataset.csv
data/raw/olist_order_reviews_dataset.csv
data/raw/olist_customers_dataset.csv
data/raw/olist_products_dataset.csv
data/raw/olist_sellers_dataset.csv
data/raw/olist_geolocation_dataset.csv
data/raw/product_category_name_translation.csv
```

### Outputs
```
processed/master_orders.csv    ~99k rows × 25 cols
```

### Key columns produced

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | str | Primary key |
| `customer_id` | str | Customer identifier |
| `order_status` | str | delivered / cancelled / etc |
| `order_purchase_timestamp` | datetime | When order was placed |
| `order_delivered_customer_date` | datetime | When order arrived |
| `order_estimated_delivery_date` | datetime | Estimated arrival |
| `order_value` | float | Sum of item prices |
| `freight_value` | float | Sum of freight charges |
| `n_items` | int | Number of items in order |
| `payment_type` | str | credit_card / boleto / etc |
| `payment_installments` | int | Number of payment instalments |
| `review_score` | float | 1–5 customer review |
| `category_english` | str | Product category (English) |
| `customer_state` | str | Brazilian state code |
| `delivery_delay_days` | float | Actual minus estimated delivery (days). Positive = late |
| `delivery_days` | float | Purchase to delivery duration (days) |
| `freight_ratio` | float | freight_value / order_value |
| `purchase_month` | int | 1–12 |
| `purchase_weekday` | int | 0=Monday … 6=Sunday |
| `purchase_hour` | int | 0–23 |
| `purchase_year` | int | 2016 / 2017 / 2018 |

### Null handling decisions

| Column | Nulls | Decision | Reason |
|--------|-------|----------|--------|
| `order_delivered_customer_date` | ~2,965 | Leave as NaT | Order not yet delivered |
| `order_approved_at` | ~160 | Leave as NaT | Cancelled orders |
| `review_score` | ~1,000 | Leave as NaN | Filtered per-notebook |
| `review_comment_*` | ~58k | Leave as NaN | Optional field, unused |
| `product_category_name` | ~610 | Fill → `'unknown'` | Required for joins |
| `product_weight_g` etc | 2 | Leave | Too few to matter |

### Customisation

**Change the save path:**
```python
SAVE_PATH = 'processed/master_orders.csv'   # line in Section 14
# change to any relative or absolute path
```

**Change the data directory:**
```python
DATA_DIR = Path('rawdata')   # line in Section 2
# update to wherever your CSVs live
```

**Add more features:**
Add derived columns in Section 11 before the `to_csv` call.
Example — add total order spend including freight:
```python
master['total_spend'] = master['order_value'] + master['freight_value']
```

---

## Notebook 02 — EDA

**File:** `02_eda.ipynb`

### What it does
Four-section exploratory analysis answering specific business questions.
All charts use Plotly (interactive). No ML, no outputs — just findings.

### Inputs
```
processed/master_orders.csv
```

### Outputs
None saved to disk. Findings written in markdown cells.

### Filtered subsets used

| Variable | Filter | Used in |
|----------|--------|---------|
| `delivered` | `order_status == 'delivered'` | All sections |
| `reviewed` | delivered + `review_score` not null | Section 2 |
| `revenue_df` | delivered + `order_value` not null | Section 1 |

### Four sections

**Section 1 — Revenue & Growth**
- Monthly GMV line chart (resampled to month-end)
- Top 15 categories by total revenue (horizontal bar, colour = avg order value)
- Orders by weekday and hour (side-by-side bar)

**Section 2 — Delivery Performance**
- Delay distribution histogram with on-time reference line
- Avg review score by delay bucket (6 buckets from "Early >7d" to ">14 days late")
- Spearman correlation between delay and review score
- 1-star rate per delay bucket
- Avg delivery days by state (bar chart, colour scale red→green)

**Section 3 — Geography**
- Order volume by state (bar, colour = avg order value)
- Freight ratio by state (bar, colour = freight burden)

**Section 4 — Seller Quality**
- Seller avg review score distribution (histogram, sellers with 10+ orders only)
- Volume vs quality scatter (size = revenue, colour = avg delay)
- Bottom 5% seller impact on 1-star reviews

### Customisation

**Change the delay buckets:**
```python
bins   = [-999, -7, 0, 3, 7, 14, 999]          # Section 2b
labels = ['Early >7d', 'On time / Early', ...]  # adjust to match
```

**Change the number of top categories shown:**
```python
.head(15)    # Section 1b — change to any number
```

**Change the seller minimum order threshold:**
```python
seller_stats[seller_stats['n_orders'] >= 10]   # Section 4b
# increase to 20+ for cleaner distribution, decrease for more sellers
```

**Export a chart as HTML (for README):**
```python
fig.write_html('outputs/delivery_delay_chart.html')
```

---

## Notebook 03 — Churn Prediction

**File:** `03_churn_model.ipynb`

### What it does
Defines churn, engineers features from each customer's first order,
handles class imbalance with SMOTE, trains Logistic Regression baseline
then XGBoost, evaluates with AUC-ROC and precision-recall,
explains predictions with SHAP.

### Inputs
```
processed/master_orders.csv
```

### Outputs
```
processed/churn_predictions.csv    customer_id, churned, churn_probability
models/churn_xgb.pkl               trained XGBoost model
```

### Churn definition
A customer is labelled **churned (1)** if they placed no second order
within 6 months of their first order.

Customers whose first order falls within the last 6 months of the dataset
are **excluded** — we cannot yet observe whether they returned.

```python
eligible['churned'] = (
    (eligible['n_orders'] == 1) |
    (eligible['last_order'] <= eligible['first_order'] + pd.DateOffset(months=6))
).astype(int)
```

### Features

| Feature | Source | Notes |
|---------|--------|-------|
| `order_value` | master | Total spend on first order |
| `freight_value` | master | Freight on first order |
| `freight_ratio` | master | freight / order_value |
| `n_items` | master | Items in first order |
| `payment_installments` | master | Max instalments used |
| `review_score` | master | Score given after first order |
| `delivery_delay_days` | master | Delay on first order |
| `delivery_days` | master | Total delivery duration |
| `purchase_month` | master | Month of first order |
| `purchase_weekday` | master | Day of week |
| `purchase_hour` | master | Hour of day |
| `purchase_year` | master | Year |
| `payment_type_enc` | encoded | Label-encoded payment method |
| `category_english_enc` | encoded | Label-encoded product category |
| `customer_state_enc` | encoded | Label-encoded state |

### Model pipeline
```
raw features
    │
    ▼
LabelEncoder  (payment_type, category_english, customer_state)
    │
    ▼
train_test_split (80/20, stratified)
    │
    ▼
SMOTE on training set only
    │
    ├──▶ LogisticRegression (baseline)
    │
    └──▶ XGBoostClassifier
             │
             ▼
         AUC-ROC / Precision-Recall / Confusion Matrix
             │
             ▼
         SHAP TreeExplainer
```

### Evaluation metrics used

| Metric | Why |
|--------|-----|
| AUC-ROC | Primary metric — threshold-independent, handles imbalance well |
| Precision-Recall | More informative than ROC for severe imbalance |
| F1 at threshold=0.3 | More aggressive recall — catches more churners |
| 5-Fold CV AUC | Confirms score is stable, not a lucky split |

### Customisation

**Change the churn window:**
```python
pd.DateOffset(months=6)   # Section 2 — change to 3 or 12 months
```

**Change the classification threshold:**
```python
threshold = 0.3   # Section 9 — lower = more churners flagged, lower precision
```

**Add or remove features:**
```python
feature_cols = [
    'order_value', 'freight_value', ...   # Section 4
    # add any numeric column from master_orders.csv here
]
```

**Tune XGBoost:**
```python
xgb_model = xgb.XGBClassifier(
    n_estimators=300,          # increase for better accuracy, slower training
    max_depth=5,               # decrease to reduce overfitting
    learning_rate=0.05,        # decrease with more estimators
    subsample=0.8,             # fraction of rows per tree
    colsample_bytree=0.8,      # fraction of features per tree
)
```

---

## Notebook 04 — Demand Forecasting

**File:** `04_demand_forecasting.ipynb`

### What it does
Aggregates daily order counts per product category, benchmarks a
7-day rolling average baseline, then fits Facebook Prophet with
Brazilian public holidays. Evaluates on a 60-day holdout.

### Inputs
```
processed/master_orders.csv
```

### Outputs
```
processed/demand_forecasts.csv    ds, yhat, yhat_lower, yhat_upper, category
```

### Prophet configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| `yearly_seasonality` | True | Captures Christmas, Black Friday patterns |
| `weekly_seasonality` | True | Captures weekday ordering patterns |
| `daily_seasonality` | False | Not enough data for reliable daily patterns |
| `changepoint_prior_scale` | 0.05 | Controls flexibility of trend changes. Higher = more flexible |
| `interval_width` | 0.80 | Width of uncertainty band (80% confidence interval) |
| `holidays` | BR public holidays | 22 holiday events 2016–2018 |

### Evaluation on holdout

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | mean(|actual − predicted|) | Avg absolute error in orders/day |
| MAPE | mean(|actual − predicted| / actual) × 100 | % error — comparable across categories |

### Customisation

**Change holdout length:**
```python
HOLDOUT_DAYS = 60   # Section 4 — increase for tougher evaluation
```

**Change number of categories to forecast:**
```python
top5_cats = (                # Section 2
    delivered['category_english']
    .value_counts()
    .head(5)                 # change to .head(10) for more categories
    .index.tolist()
)
```

**Add more holidays:**
```python
br_holidays = pd.DataFrame({
    'holiday': [...],   # Section 6 — add any event name
    'ds':      [...],   # add the date
    'lower_window': 0,  # days before holiday to include
    'upper_window': 1,  # days after holiday to include
})
```

**Make Prophet more/less flexible:**
```python
changepoint_prior_scale=0.05   # default: conservative, stable trend
changepoint_prior_scale=0.3    # more flexible — follows data more closely
                               # risk: overfitting to noise
```

**Forecast further ahead:**
```python
future = m.make_future_dataframe(periods=holdout_days)
# change holdout_days to 90 or 180 for longer-range forecast
```

---

## Notebook 05 — Customer Segmentation

**File:** `05_segmentation.ipynb`

### What it does
Computes RFM (Recency, Frequency, Monetary) scores per customer,
log-transforms skewed features, scales with StandardScaler,
uses elbow + silhouette to find optimal k, fits K-Means,
assigns business labels, and connects segment labels to
churn predictions from notebook 03.

### Inputs
```
processed/master_orders.csv
processed/churn_predictions.csv     (from notebook 03)
```

### Outputs
```
processed/customer_segments.csv      customer_id, recency, frequency, monetary,
                                     cluster, segment, pca1, pca2
processed/churn_with_segments.csv    all churn_predictions cols + segment info
models/kmeans_segmentation.pkl       fitted KMeans model
models/rfm_scaler.pkl                fitted StandardScaler
```

### RFM definitions

| Feature | Definition | Direction |
|---------|------------|-----------|
| Recency | Days since last order (from dataset end date) | Lower = better |
| Frequency | Total number of orders placed | Higher = better |
| Monetary | Total spend (R$) across all orders | Higher = better |

### Why log-transform before scaling?

Frequency and Monetary are right-skewed — a small number of customers
have very high values that dominate K-Means distance calculations.
`log1p` compression brings the distribution closer to normal
before StandardScaler normalises to mean=0, std=1.

```python
rfm['log_frequency'] = np.log1p(rfm['frequency'])
rfm['log_monetary']  = np.log1p(rfm['monetary'])
features = ['recency', 'log_frequency', 'log_monetary']
```

### Segment labelling guide

After running Section 6, read the cluster profile table and assign
labels using this guide:

| Profile | Business Label |
|---------|---------------|
| Low recency + high frequency + high monetary | Champions |
| Low recency + medium frequency + medium monetary | Loyal Customers |
| Medium/high recency + was previously active | At Risk |
| Any recency + frequency = 1 + low monetary | One-Time Buyers |
| Very low recency + frequency = 1 | New Customers |

**Update the label_map in Section 7 to match your actual cluster numbers:**
```python
label_map = {
    0: 'Champions',        # update these numbers
    1: 'Loyal Customers',  # to match your profile table
    2: 'At Risk',
    3: 'One-Time Buyers',
    4: 'New Customers',
}
```

### Customisation

**Change the number of clusters:**
```python
K = 5   # Section 6 — read elbow + silhouette plots before changing
        # k=4 gives broader segments, k=6 gives finer ones
        # always re-run label_map after changing K
```

**Change RFM reference date:**
```python
ref_date = delivered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
# default: day after last order in dataset
# for production use: ref_date = pd.Timestamp.today()
```

**Add extra features to clustering:**
```python
# merge additional features from master before scaling
rfm = rfm.merge(
    delivered.groupby('customer_id')['review_score'].mean().reset_index(),
    on='customer_id', how='left'
)
features = ['recency', 'log_frequency', 'log_monetary', 'review_score']
```

**Use DBSCAN instead of K-Means:**
```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=50)
rfm['cluster_dbscan'] = db.fit_predict(X_scaled)
# note: DBSCAN assigns -1 to noise points — handle these separately
# n_clusters = len(set(rfm['cluster_dbscan'])) - (1 if -1 in rfm['cluster_dbscan'] else 0)
```

**Predict segment for a new customer:**
```python
import joblib
scaler = joblib.load('models/rfm_scaler.pkl')
km     = joblib.load('models/kmeans_segmentation.pkl')

new_customer = {
    'recency':        45,
    'log_frequency':  np.log1p(2),
    'log_monetary':   np.log1p(250),
}
X_new   = scaler.transform([[new_customer[f] for f in features]])
cluster = km.predict(X_new)[0]
segment = label_map[cluster]
print(f'Predicted segment: {segment}')
```

---

## Cross-Notebook Dependencies

| Notebook | Reads | Writes |
|----------|-------|--------|
| 01 | 9 raw CSVs | `master_orders.csv` |
| 02 | `master_orders.csv` | — |
| 03 | `master_orders.csv` | `churn_predictions.csv`, `churn_xgb.pkl` |
| 04 | `master_orders.csv` | `demand_forecasts.csv` |
| 05 | `master_orders.csv`, `churn_predictions.csv` | `customer_segments.csv`, `churn_with_segments.csv`, `kmeans_segmentation.pkl`, `rfm_scaler.pkl` |

**If you re-run notebook 01** (e.g. after changing features):
- Re-run 02 to update EDA charts
- Re-run 03 to retrain churn model on new features
- Re-run 05 after 03 to update churn_with_segments

**If you re-run notebook 03** (e.g. after tuning XGBoost):
- Re-run 05 to update cross-model connection

Notebooks 02 and 04 are independent of each other and can be re-run at any time.

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: master_orders.csv` | Notebook 01 not run yet or wrong path | Run 01 first; check `SAVE_PATH` variable |
| `FileNotFoundError: churn_predictions.csv` | Notebook 03 not run yet | Run 03 before 05 |
| `KeyError: 'category_english'` | Running 02/03/04/05 before 01 | Run 01 first |
| `ArrowKeyError` on `to_parquet` | pyarrow not installed | Use `to_csv` instead (already default) |
| Prophet install fails on Windows | Compiler issue | Use `conda install -c conda-forge prophet` |
| SMOTE `ValueError` | Only one class in training fold | Check churn label has both 0 and 1 values |
| SHAP `TypeError` | Mismatched feature names | Ensure `X_test` has same columns as training |
| K-Means gives 1 large cluster | Features not scaled | Confirm StandardScaler applied before KMeans |

---

## Extending the Project

### Add a sixth notebook — combined insights

Create `06_insights_report.ipynb` that loads all processed outputs and
produces a single summary: top churn drivers × segment, forecast vs actual
per segment, seller quality heatmap.

### Deploy the Dash dashboard

```bash
pip install dash dash-bootstrap-components gunicorn
# add app.py (see dashboard module)
# deploy to Render as a Web Service pointing at app.py
```

### Add MLflow tracking

```python
import mlflow
with mlflow.start_run(run_name="xgb_churn_v1"):
    mlflow.log_param("n_estimators", 300)
    mlflow.log_metric("auc_roc", xgb_auc)
    mlflow.sklearn.log_model(xgb_model, "model")
```

### Add Optuna hyperparameter tuning

```python
import optuna

def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 8),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train_bal, y_train_bal,
                            cv=3, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('Best params:', study.best_params)
```
