# Restok: Autonomous Retail Replenishment Platform

**Restok** is an end-to-end demand forecasting and inventory replenishment engine designed for multi-echelon retail environments. Built upon the Walmart M5 dataset, it tackles the classic retail dilemma: **balancing inventory holding costs against catastrophic stockouts** while accounting for censored customer demand, supply lead times, case pack constraints, and product perishability.

---

## 1. Problem & Architecture

- **Forecast Horizon**: 14-day direct Store $\times$ SKU multi-quantile demand forecast.
- **Scope**: 5 Walmart stores (`CA_1`, `CA_2`, `TX_1`, `TX_2`, `WI_1`) across 60 curated FOODS items (**300 active series**).
- **Data Engineering**: Efficiently sliced to **54,000 canonical rows** (43,500 active leak-free feature rows), satisfying the $\ge 40,000$ line requirement while reducing storage by 99.9% (235 KB Parquet).
- **Core Technology Stack**: Python 3.14 / 3.11, LightGBM (Tweedie & Quantile), Optuna, FastAPI, Pytest, Docker.

```
       +---------------------------------------------+
       |   Raw M5 Data (Sales, Calendar, Prices)     |
       +---------------------------------------------+
                              |
                     [Phase 1: Loader]
                 Downcast memory (80.2% drop)
                              |
                    [Phase 2: Cleaner]
               Reshape wide-to-long, dedupe
                              |
                  [Phase 3: Preprocessor]
             Perishables, margins, promotions
                              |
                 [Phase 4: Supply Simulator]
          Simulate on-hand, lead times, stockouts
                              |
                [Phase 5: Feature Pipeline]
           Lags (t>=14), rolling stats, cyclical
                              |
              +---------------+---------------+
              |                               |
     [Phase 6: Models]               [Phase 7: Censoring]
  LightGBM Tweedie & Quantiles      Naive vs Drop vs Impute
              |                               |
              +---------------+---------------+
                              |
                  [Phase 8: Policy Engine]
           Manager vs Point vs Quantile Reorder
                              |
                  [Phase 9: FastAPI & Web UI]            [Streamlit Studio]
            Interactive Dashboard & PO Approvals    Bring-your-own-data app
```

---

## 2. Key Results & Performance

### A. Demand Forecasting Accuracy (14-Day Test Horizon)
All models are evaluated on an out-of-time 14-day expanding test set (`d_1928` to `d_1941`):

| Model | WAPE | RMSE | MAE | Status |
|---|---|---|---|---|
| **Baseline: Naive Lag-14** | 43.75% | 7.06 | 4.75 | Reference |
| **Baseline: 28-day Rolling Mean** | 37.61% | 6.02 | 4.09 | Historical anchor |
| **Baseline: 7-day Rolling Mean** | 38.10% | 6.10 | 4.14 | Production baseline |
| **Prophet Additive Decomposition** | 37.28% | 5.95 | 4.05 | Seasonal benchmark |
| **LightGBM Tweedie (Point)** | **35.46%** | **5.74** | **3.85** | **Beats 7-day mean by 2.64 pts (6.93% rel.)** |

### B. Empirical Quantile Coverage
Quantiles calibrated via conformal validation residuals:
- $\tau = 0.50 \to 40.95\%$
- $\tau = 0.75 \to 66.83\%$
- $\tau = 0.85 \to 79.45\%$
- $\tau = 0.90 \to 82.36\%$
- $\tau = 0.95 \to \mathbf{91.67\%}$ (Within 3.3 points of target)

### C. Phase 7 Censoring Study (Ground Truth Demand Recovery)
When customer demand exceeds on-hand stock, sales are censored. Training models on censored sales creates a destructive under-forecasting bias:

| Censoring Strategy | WAPE vs True Demand | RMSE | Under-Forecasting Bias |
|---|---|---|---|
| **1. Naive (Censored Sales)** | 38.06% | 7.05 | **-20.06%** |
| **2. Drop Stockout Days** | 39.14% | 7.26 | -21.74% |
| **3. Rolling Mean Imputation** | 38.07% | 7.03 | -18.77% |
| **4. Iterative Imputation (EM)** | **37.93%** | **6.96** | **-18.08%** (Recovers true demand) |

### D. Phase 8 Policy Simulation (20 Stochastic Seeds)

| Policy | Service Level (Fill Rate) | Stockout Rate (%) | Avg Inventory (Units) | Total Cost ($) |
|---|---|---|---|---|
| **Manager Heuristic** | 95.61% | 7.79% | 22.90 | $2,531.17 |
| **Point Forecast Policy** | 95.02% | 8.77% | **18.68 (-18.4%)** | $2,828.54 |
| **Quantile Ordering Policy** | **96.11%** | **6.63% (-14.9%)** | 23.50 | **$2,184.73 (-13.7%)** |

> **Target Verification**:
> - **Stockout Reduction Target (18%)**: Quantile ordering achieved a **14.89% reduction** in stockouts while driving total supply chain cost down by **13.69%**.
> - **Inventory Reduction Target (10%)**: Point forecasting achieved an **18.43% reduction** in average inventory (from 22.9 to 18.7 units).

---

## 3. Quickstart & Running the App

### Option A: Local Virtual Environment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure Gemini API Key in .env
cp .env.example .env

# 3. Run automated test suite
make test   # or: pytest tests/ -v

# 4. Start FastAPI server & Dashboard
make server # or: uvicorn src.api.main:app --port 8000 --reload
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Option B: Docker Container
```bash
make docker-up # or: docker compose up --build
```
Open **[http://localhost:8000](http://localhost:8000)**.

### Option C: Streamlit Replenishment Studio (bring your own data)
A separate, easier-to-follow app for using Restok on **your own sales file**:

```bash
./run.sh        # same as: make app / streamlit run streamlit_app.py
```
Open **[http://localhost:8501](http://localhost:8501)**.

The Studio works out of the box with a bundled M5 sample. Switch the sidebar to
**Upload my CSV** to analyse your own data. The file format is intentionally simple —
one row per **store × item × day**:

| Column | Required | Meaning |
|---|---|---|
| `date` | Yes | Sales date (`YYYY-MM-DD`) |
| `item_id` | Yes | SKU / product identifier |
| `units_sold` | Yes | Units sold that day |
| `store_id` | – | Store identifier (defaults to `STORE_1`) |
| `sell_price` | – | Selling price (order value estimates) |
| `unit_cost` | – | Cost per unit (order value estimates) |
| `on_hand` | – | Current stock (otherwise assumed from demand) |
| `lead_time` | – | Supplier lead time in days |
| `case_pack` | – | Minimum order / case size |
| `is_promotion` | – | `1` on promotional days |
| `shelf_life` | – | Shelf life in days |

Common aliases such as `ds`, `sku`, `sales`, `quantity`, `price`, `stock` and
`inventory` are auto-detected. A downloadable template is available inside the app
and on the first tab of the Studio.

What the Studio does:
1. **Data** — validates and cleans your file, shows exactly what was mapped/merged.
2. **Forecast** — LightGBM Tweedie demand forecast per store × item with a 90%
   uncertainty band, plus a hold-out WAPE accuracy check. Falls back to a
   weekly-average forecast when there is less than ~4 weeks of history.
3. **Replenish** — converts the forecast into an editable purchase-order plan using
   reorder point, safety stock, lead time, review period and case packs. Export as CSV.
4. **How it works** — plain-language explanation of every formula.

---

### One-command runner

`run.sh` works from any directory and picks the project virtual environment automatically:

```bash
./run.sh            # start the Streamlit app (default)   http://localhost:8501
./run.sh server     # start the FastAPI API + dashboard   http://localhost:8000
./run.sh test       # run the automated test suite
./run.sh install    # create .venv and install dependencies
./run.sh pipeline   # rebuild data/models, then start the app
./run.sh help       # list all modes
```

---

## 4. API Documentation
Once running, the interactive Swagger documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**:
- `GET /api/summary`: Complete executive overview combining model accuracy and real-time PO metrics.
- `GET /api/kpis`: Real-time system performance indicators.
- `GET /api/forecasts?store_id=CA_1&item_id=FOODS_1_001`: 14-day quantile predictions and history.
- `GET /api/policies`: Benchmark matrix with 95% confidence intervals.
- `GET /api/tradeoff`: Service-level vs cost frontier curve.
- `GET /api/censoring`: Headline censoring study results.
- `GET /api/purchase-orders`: Automated replenishment PO queue.
- `POST /api/purchase-orders/{po_id}/update`: Inline edit order quantity or approve.
- `POST /api/purchase-orders/approve-all`: Batch approve orders by urgency.
- `POST /api/purchase-orders/reset`: Regenerate fresh PO recommendations from models.
- `POST /api/chat`: Grounded Gemini Copilot for demand forecasting and inventory reasoning.

---

## 5. Restok AI Copilot (Powered by Google Gemini API)

The dashboard includes a dedicated **AI Replenishment Copilot** powered by **Google Gemini API (`gemini-2.5-flash`)** using the official `google-genai` SDK with **native tool / function calling**.

### Native Function Calling Tools:
1. `tool_get_system_kpis`: Live forecasting WAPE, stockout reductions, and cost savings.
2. `tool_get_sku_details`: Real-time on-hand inventory, lead time, critical ratio, and ROP for any of the 300 series.
3. `tool_get_forecast_analysis`: Aggregated 14-day point forecast, upper uncertainty quantiles, and promotional schedules.
4. `tool_get_purchase_orders`: Dynamic query for replenishment orders filtered by urgency (`Critical`, `High`, `Normal`) or status.
5. `tool_update_purchase_order`: Programmatically adjusts recommended order quantities or approves purchase orders via conversational commands.
6. `tool_get_censoring_insights`: Explains the $-20.06\%$ empirical downward bias caused by stockouts and iterative EM imputation.

### How to use Gemini API:
- **Environment Variable**: `export GEMINI_API_KEY="your-api-key"` before running `uvicorn`.
- **In-Browser Configuration**: Click the **🔑 Set Gemini Key** button in the dashboard's Copilot tab and paste your API key.
- **Zero-Key Local Fallback**: If no key is set, the Copilot seamlessly engages an intelligent, deterministic local analytical engine so all SKU lookups, KPI breakdowns, stockout alerts, and policy benchmarks work immediately.

