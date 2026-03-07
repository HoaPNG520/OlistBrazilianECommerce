# Project Notes & Experiment Log

> One entry per day. Write findings immediately after running — don't reconstruct from memory later.
> These notes feed directly into your README and CV bullets.

---

## Day 1 — Ingestion & Cleaning

**Date:** ___________

**Data loaded:**
| Table | Rows | Notes |
|-------|------|-------|
| orders | | |
| order_items | | |
| order_payments | | |
| order_reviews | | |
| customers | | |
| products | | |
| sellers | | |
| geolocation | → sampled to unique zips | |
| categories | | |

**master_orders.csv shape:** _____ rows × _____ cols

**Null decisions:**
- `order_delivered_customer_date`: _____ nulls — left as NaT (undelivered orders)
- `review_comment_message`: _____ nulls — left (optional field, unused)
- `product_category_name`: _____ nulls — filled with 'unknown'
- Other: ___________

**Issues hit & how resolved:**
-
-

**Output saved:** `processed/master_orders.csv`

---

## Day 2 — EDA

**Date:** ___________

### Section 1 — Revenue & Growth
- Total GMV: R$ ___________
- Date range: ___________ to ___________
- Peak month: ___________ (R$ ___________)
- Growth (first → last month): ___________%
- Top 3 categories by GMV: ___________, ___________, ___________
- **Finding:** ___________________________________________________________

### Section 2 — Delivery Performance
- % orders delivered on time or early: ___________%
- % orders delivered late: ___________%
- Avg delay for late orders: _____ days
- Spearman r (delay vs review score): _____ (p=_______)
- 1-star rate for on-time orders: ___________%
- 1-star rate for >14 day late orders: ___________%
- Fastest state: _____ (avg _____ days)
- Slowest state: _____ (avg _____ days)
- **Finding:** ___________________________________________________________

### Section 3 — Geography
- SP share of total orders: ___________%
- Top 3 states by order volume: ___________, ___________, ___________
- Highest freight burden state: _____ (_____ % of order value)
- Lowest freight burden state: _____ (_____ % of order value)
- **Finding:** ___________________________________________________________

### Section 4 — Seller Quality
- Total sellers: ___________
- Sellers with avg review < 3.0: _____ (_____%)
- Bottom 5% sellers share of orders: ___________%
- Bottom 5% sellers share of 1-star reviews: ___________%
- **Finding:** ___________________________________________________________

**Issues hit & how resolved:**
-
-

---

## Day 3 — Churn Model

**Date:** ___________

**Churn definition:** No return purchase within 6 months of first order

**Dataset:**
- Eligible customers: ___________
- Churned (1): _____ (_____%)
- Retained (0): _____ (_____%)
- Class imbalance ratio: _____:1

**Features used:** ___________________________________________________________

**Results:**
| Metric | Logistic Regression | XGBoost |
|--------|--------------------|---------| 
| AUC-ROC | | |
| Precision (churn, t=0.3) | | |
| Recall (churn, t=0.3) | | |
| F1 (churn, t=0.3) | | |
| 5-Fold CV AUC | | |

**Top SHAP features (churn drivers):**
1. ___________
2. ___________
3. ___________
4. ___________
5. ___________

**Key observations:**
-
-

**Output saved:** `processed/churn_predictions.csv`, `models/churn_xgb.pkl`

**Issues hit & how resolved:**
-
-

---

## Day 4 — Demand Forecasting

**Date:** ___________

**Top 5 categories:** ___________, ___________, ___________, ___________, ___________

**Holdout period:** last _____ days

**Black Friday check:** _____ orders on 2017-11-24 (_____ × average)

**Results:**
| Category | Baseline MAPE | Prophet MAPE | Improvement |
|----------|--------------|-------------|-------------|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Seasonality observed:**
- Weekly: ___________________________________________________________
- Yearly: ___________________________________________________________
- Holidays: ___________________________________________________________

**Key observations:**
-
-

**Output saved:** `processed/demand_forecasts.csv`

**Issues hit & how resolved:**
-
-

---

## Day 5 — Customer Segmentation

**Date:** ___________

**RFM summary:**
- Total customers: ___________
- Avg recency: _____ days
- Avg frequency: _____ orders
- Avg monetary: R$ _____

**Optimal k chosen:** _____ (silhouette score: _____)

**Segment profiles:**
| Segment | Label | Customers | Avg Recency | Avg Frequency | Avg Monetary | Avg Churn Prob |
|---------|-------|-----------|-------------|---------------|--------------|----------------|
| 0 | | | | | | |
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |

**Cross-model finding (segment × churn):**
- Highest churn probability segment: ___________
- Lowest churn probability segment: ___________
- Gap between highest and lowest: ___________%

**PCA explained variance:** PC1=_____%, PC2=_____%

**Key observations:**
-
-

**Outputs saved:** `processed/customer_segments.csv`, `processed/churn_with_segments.csv`,
`models/kmeans_segmentation.pkl`, `models/rfm_scaler.pkl`

**Issues hit & how resolved:**
-
-

---

## Backlog / Ideas for Extension

- [ ] Deploy Dash dashboard to Render
- [ ] Add DBSCAN as alternative segmentation comparison
- [ ] Tune XGBoost hyperparameters with Optuna
- [ ] Add confidence intervals to churn predictions
- [ ] Forecast at weekly grain for smoother Prophet results
- [ ] Add a sixth notebook: combined insights report

---

## Final Numbers for CV Bullets

*(Fill this in last — copy from daily entries above)*

- Churn AUC-ROC: ___________
- Churn top SHAP driver: ___________
- Best Prophet MAPE: ___________%  vs baseline ___________%
- Segmentation silhouette: ___________
- Churn probability gap (best vs worst segment): ___________%
- SP GMV share: ___________%
- Bottom 5% sellers → ___% of 1-star reviews
- Delivery delay Spearman r: ___________
