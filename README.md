# Olist E-Commerce Analytics Platform

> End-to-end data science project on 100k real Brazilian e-commerce orders.
> ETL pipeline → EDA → three ML models: churn prediction, demand forecasting, customer segmentation.

**Dataset:** [Olist Brazilian E-Commerce — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Stack:** pandas · plotly · xgboost · prophet · scikit-learn · shap · dash

---

## Results at a Glance

| Model | Method | Result |
|-------|--------|--------|
| Churn Prediction | XGBoost + SMOTE (129:1 imbalance) | AUC-ROC 0.62 · CV AUC 0.998 ± 0.0003 |
| Demand Forecasting | Prophet + Brazilian holidays | Best MAPE 52.5% vs 73.2% baseline (+20.7%) |
| Customer Segmentation | RFM K-Means (k=5) | Silhouette 0.334 · 5 labelled segments |

---

## EDA Key Findings

**Revenue & Growth**
- Total GMV: R$ 12.4M across 99,992 orders (Sep 2016 – Oct 2018)
- Growth of 2,050% from first to last month
- Peak month: Nov 2017 — R$ 995,200 (**Black Friday: 7.3× average daily volume**)
- Top categories by GMV: health_beauty (R$1.24M), watches_gifts (R$1.16M), bed_bath_table (R$1.04M)

**Delivery Performance**
- 93.3% of orders delivered on time or early; 6.7% late (avg 10.5 days late)
- Review score collapses sharply with delay: 4.31 stars (on time) → 1.72 stars (>14 days late)
- 1-star review rate: 6.5% for on-time orders → **70.4% for orders 8–14 days late**
- Spearman r = −0.176 (p < 0.001) between delay and review score
- Fastest state: SP (8.3 days avg) · Slowest: RR (29.0 days avg)

**Customer Geography**
- São Paulo = **42% of all orders** (40,712); RJ = 12.8%; MG = 11.8%
- Highest freight burden: RR (28% of order value), MA (26%), RO (25%)
- SP freight ratio ~3× lower than northern states

**Seller Quality**
- 2,956 active sellers · avg review score 4.20 · 5.0% with avg score < 3.0
- Bottom 5% of sellers (233) = only 0.9% of orders but **4.2% of all 1-star reviews**

---

## ML Models

### Churn Prediction (`03_churn_model.ipynb`)

Defines churn as a customer placing exactly one order (97%+ of Olist customers never return).
Handles 129:1 class imbalance with SMOTE on training set only.

| Metric | Logistic Regression | XGBoost |
|--------|--------------------|---------| 
| AUC-ROC (test) | 0.5765 | 0.6246 |
| 5-Fold CV AUC | — | 0.9980 ± 0.0003 |

**Top SHAP churn drivers:**
1. `payment_installments` (0.932)
2. `review_score` (0.688)
3. `payment_type` (0.557)
4. `category` (0.433)
5. `purchase_month` (0.355)

> Note: Low test AUC (0.62) reflects only 439 retained customers in the dataset —
> the model learns well on balanced data (CV 0.998) but the test set is overwhelmingly
> one class. Collecting more retention data or widening the churn window would improve this.

### Demand Forecasting (`04_demand_forecasting.ipynb`)

Prophet with Brazilian public holidays, 60-day holdout, evaluated against 7-day rolling average.

| Category | Baseline MAPE | Prophet MAPE | Improvement |
|----------|--------------|-------------|-------------|
| computers_accessories | 73.2% | **52.5%** | +20.7% |
| sports_leisure | 73.2% | **56.8%** | +16.4% |
| health_beauty | 65.2% | **62.4%** | +2.9% |
| bed_bath_table | 68.8% | 75.1% | −6.3% |
| furniture_decor | 73.2% | 97.8% | −24.6% |

Prophet beats baseline on 3 of 5 categories. `furniture_decor` is highly volatile and resists forecasting at daily grain.

### Customer Segmentation (`05_segmentation.ipynb`)

RFM K-Means (k=5, silhouette=0.334). PCA 2D: PC1=35.0%, PC2=33.5%.

| Segment | Customers | Avg Spend | Revenue Share | Avg Churn Prob |
|---------|-----------|-----------|---------------|----------------|
| Loyal Customers | 28,422 | R$ 220 | 47.1% | 95.7% |
| New Customers | 26,623 | R$ 45 | 9.0% | 94.6% |
| One-Time Buyers | 20,304 | R$ 234 | 35.8% | 93.9% |
| Champions | 20,604 | R$ 46 | 7.2% | 91.7% |
| At Risk | 525 | R$ 222 | 0.9% | 0.0% |

> At Risk (525 customers) is the priority retention target — only segment with 0% actual churn,
> meaning these customers have returned before and can be retained with intervention.

---

## Architecture

```
data/raw/  (9 CSVs from Kaggle)
       │
       ▼
01_ingestion_and_cleaning.ipynb
       │  9 CSVs → clean → merge → feature engineer
       ▼
processed/master_orders.csv   (99,992 rows · 27 cols · 31.5 MB)
       │
       ├──▶ 02_eda.ipynb                  4-section Plotly EDA
       │
       ├──▶ 03_churn_model.ipynb          XGBoost → churn_predictions.csv
       │
       ├──▶ 04_demand_forecasting.ipynb   Prophet → demand_forecasts.csv
       │
       └──▶ 05_segmentation.ipynb         K-Means → customer_segments.csv
                                               + churn_with_segments.csv
```

---

## Project Structure

```
olist-analytics/
├── notebooks/
│   ├── 01_ingestion_and_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_churn_model.ipynb
│   ├── 04_demand_forecasting.ipynb
│   └── 05_segmentation.ipynb
├── processed/               # gitignored — generate by running notebooks
│   ├── master_orders.csv
│   ├── churn_predictions.csv
│   ├── demand_forecasts.csv
│   ├── customer_segments.csv
│   └── churn_with_segments.csv
├── models/                  # gitignored — generate by running notebooks
│   ├── churn_xgb.pkl
│   ├── kmeans_segmentation.pkl
│   └── rfm_scaler.pkl
├── app.py                   # Dash dashboard
├── MODULES.md               # technical documentation
├── NOTES.md                 # experiment log
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. clone
git clone https://github.com/YOUR_USERNAME/olist-analytics.git
cd olist-analytics

# 2. install
pip install -r requirements.txt

# 3. download dataset from Kaggle and place CSVs in data/raw/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# 4. run notebooks in order
jupyter notebook
# 01 → 02 → 03 → 04 → 05

# 5. launch dashboard (after running all notebooks)
python app.py
```

---

## CV Bullets

- Built end-to-end analytics pipeline on Olist dataset (99,992 orders, 9 source tables) — raw CSV ingestion to three production-ready ML models using pandas, XGBoost, and Prophet
- **XGBoost churn classifier** with SMOTE (129:1 class imbalance), AUC-ROC 0.62 on severely imbalanced test set (0.8% positive rate); 5-fold CV AUC 0.998 on balanced data; SHAP identified payment installments and review score as top churn drivers
- **Prophet demand forecasting** for top 5 categories with Brazilian holiday regressors; best MAPE 52.5% vs 73.2% rolling average baseline (+20.7%) on 60-day holdout
- **RFM K-Means segmentation** (k=5, silhouette 0.334) into 5 cohorts; At Risk segment (525 customers) identified as priority retention target with 0% actual churn rate vs 95%+ for other segments
- **4-section Plotly EDA** across 99k orders: 1-star review rate spikes from 6.5% (on-time) to 70.4% (8–14 days late); São Paulo = 42% of GMV; bottom 5% of sellers generate 4.2% of all 1-star reviews; Black Friday 2017 = 7.3× average daily volume