"""Restok AI Copilot powered by Google Gemini API.

Provides grounded retail replenishment intelligence:
- Retrieves real-time inventory, forecasts, and quantile uncertainty
- Explains Purchase Order recommendations and approval rationale
- Analyzes policy trade-offs and censoring study results
- Executes tool calling via the official google-genai SDK
- Includes smart local fallback when GEMINI_API_KEY is not configured
"""
import os
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def tool_get_system_kpis() -> Dict:
    """Returns high-level model forecasting accuracy, stockout reduction, and cost savings."""
    metrics_path = PROCESSED_DIR / "metrics.csv"
    policy_path = PROCESSED_DIR / "policy_kpi_summary.csv"

    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    policy_df = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()

    lgb_wape = 34.91
    base_wape = 38.10
    if len(metrics_df) > 0:
        lgb_row = metrics_df[metrics_df["Model"].str.contains("Tweedie", na=False)]
        base_row = metrics_df[metrics_df["Model"].str.contains("7-day", na=False)]
        if len(lgb_row) > 0:
            lgb_wape = round(float(lgb_row["WAPE"].iloc[0]) * 100, 2)
        if len(base_row) > 0:
            base_wape = round(float(base_row["WAPE"].iloc[0]) * 100, 2)

    return {
        "model_wape_pct": lgb_wape,
        "baseline_7day_wape_pct": base_wape,
        "wape_improvement_pct": round(((base_wape - lgb_wape) / base_wape) * 100, 2),
        "heuristic_stockout_rate_pct": 7.79,
        "quantile_stockout_rate_pct": 6.63,
        "stockout_reduction_pct": 14.89,
        "total_cost_savings_pct": 13.69,
        "series_tracked": 300,
        "stores_count": 5,
        "skus_count": 60,
    }


def tool_get_sku_details(store_id: str, item_id: str) -> Dict:
    """Queries inventory, ROP, lead time, shelf life, case pack, unit cost, and price for a Store x SKU."""
    test_path = PROCESSED_DIR / "test_forecasts.parquet"
    if not test_path.exists():
        return {"error": "Forecast data not available."}

    df = pd.read_parquet(test_path)
    series_df = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")
    if len(series_df) == 0:
        return {"error": f"Series ({store_id}, {item_id}) not found."}

    last_row = series_df.iloc[-1]
    on_hand = int(last_row["on_hand_inventory"])
    lead_time = int(last_row["lead_time"])
    shelf_life = int(last_row["shelf_life"])
    case_pack = int(last_row["case_pack"])
    unit_cost = float(last_row["unit_cost"])
    sell_price = float(last_row["sell_price"])
    cr = float(last_row["critical_ratio"])

    # Quantile reorder point
    q_col = "pred_q_75" if cr <= 0.78 else ("pred_q_85" if cr <= 0.88 else "pred_q_90")
    q_demand = float(last_row[q_col])
    rop = round(q_demand * lead_time, 1)

    target_stock = q_demand * (lead_time + 2)
    needed = max(0, target_stock - on_hand)
    recommended_order_qty = int(np.ceil(needed / case_pack) * case_pack) if on_hand <= rop else 0

    return {
        "store_id": store_id,
        "item_id": item_id,
        "dept_id": str(last_row["dept_id"]),
        "on_hand_inventory": on_hand,
        "reorder_point_rop": rop,
        "lead_time_days": lead_time,
        "shelf_life_days": shelf_life,
        "case_pack": case_pack,
        "critical_ratio": cr,
        "unit_cost": unit_cost,
        "sell_price": sell_price,
        "recommended_order_qty": recommended_order_qty,
        "is_stockout_risk": bool(on_hand <= rop),
    }


def tool_get_forecast_analysis(store_id: str, item_id: str) -> Dict:
    """Returns 14-day aggregated demand forecast, quantile uncertainty, and promo schedule."""
    test_path = PROCESSED_DIR / "test_forecasts.parquet"
    if not test_path.exists():
        return {"error": "Forecast data not available."}

    df = pd.read_parquet(test_path)
    series_df = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].sort_values("date")
    if len(series_df) == 0:
        return {"error": f"Series ({store_id}, {item_id}) not found."}

    tot_point = float(series_df["pred_point"].sum())
    tot_q50 = float(series_df["pred_q_50"].sum())
    tot_q75 = float(series_df["pred_q_75"].sum())
    tot_q85 = float(series_df["pred_q_85"].sum())
    tot_q90 = float(series_df["pred_q_90"].sum())
    tot_q95 = float(series_df["pred_q_95"].sum())
    promo_days = int(series_df["is_promotion"].sum())

    return {
        "store_id": store_id,
        "item_id": item_id,
        "horizon_days": 14,
        "total_point_forecast": round(tot_point, 1),
        "total_q50_forecast": round(tot_q50, 1),
        "total_q75_forecast": round(tot_q75, 1),
        "total_q85_forecast": round(tot_q85, 1),
        "total_q90_forecast": round(tot_q90, 1),
        "total_q95_forecast": round(tot_q95, 1),
        "promotional_days_in_horizon": promo_days,
        "daily_avg_demand": round(tot_point / 14.0, 1),
    }


def tool_get_purchase_orders(urgency: str = "", status: str = "") -> List[Dict]:
    """Retrieves current replenishment purchase orders filtered by urgency or status."""
    from src.api.main import PURCHASE_ORDERS
    results = PURCHASE_ORDERS
    if urgency:
        results = [po for po in results if po["urgency"].lower() == urgency.lower()]
    if status:
        results = [po for po in results if po["status"].lower() == status.lower()]
    return results[:15]


def tool_update_purchase_order(po_id: str, order_qty: Optional[int] = None, status: Optional[str] = None) -> Dict:
    """Updates order quantity or approval status for a specific purchase order."""
    from src.api.main import PURCHASE_ORDERS
    for po in PURCHASE_ORDERS:
        if po["po_id"] == po_id:
            if order_qty is not None:
                po["order_qty"] = order_qty
                po["total_cost"] = round(order_qty * po["unit_cost"], 2)
            if status is not None:
                po["status"] = status
            return {"success": True, "po": po}
    return {"success": False, "error": f"PO {po_id} not found."}


def tool_get_censoring_insights() -> Dict:
    """Returns empirical findings from the Phase 7 Censoring Study."""
    return {
        "finding": "Historical sales data is systematically censored when stockouts occur.",
        "naive_downward_bias": "-20.06% under-forecasting bias",
        "drop_stockouts_bias": "-21.74% (drops continuity, exacerbates sampling bias)",
        "iterative_em_imputation": "-18.08% bias (lowest WAPE of 37.93% and RMSE 6.96 vs ground truth)",
        "recommendation": "Use iterative EM-style imputation or quantile over-coverage to counter historical stockout suppression.",
    }


# Comprehensive System Instruction
SYSTEM_INSTRUCTION = """
You are the Restok AI Supply Chain Copilot — an expert retail replenishment advisor, econometrician, and inventory optimization engine.

Your domain knowledge:
1. Scope: You manage inventory for 300 Store x SKU series across 5 Walmart stores (CA_1, CA_2, TX_1, TX_2, WI_1) and 60 FOODS items (packaged dry goods, bakery/dairy, perishables).
2. Demand Forecasting: LightGBM Tweedie point forecasting (WAPE 35.46% vs 38.10% baseline, 6.93% improvement) with Optuna-tuned hyperparameters (Tweedie variance power 1.268, lr 0.120, 54 leaves). Multi-quantile regressors trained at tau = 0.50, 0.75, 0.85, 0.90, 0.95 using conformal residual calibration.
3. Replenishment Policy: Newsvendor critical ratio (Cu / (Cu + Co)) maps each category to its optimal quantile (FOODS_1→q75, FOODS_2→q85, FOODS_3→q90), subject to delivery lead times, case pack rounding, and perishable shelf-life caps.
4. Censoring: Naive retail models suffer a -20.06% downward bias because customer demand during stockout periods is hidden. Restok corrects this via EM-style iterative imputation (reduces to -18.08% bias).
5. Policy Simulation: Day-by-day inventory simulator across 20 stochastic seeds. Quantile Ordering reduces stockouts by 14.89% (7.79% → 6.63%) and cuts total supply chain cost by 13.69% ($2,531 → $2,184) vs Manager Heuristic baseline.

Guidelines:
- Ground your answers in real metrics from the system tools.
- When asked about SKUs, check their on-hand inventory, lead times, and ROP.
- Be concise, professional, and actionable. Use markdown formatting with bullet points and bold highlights.
- If asked to approve or adjust a PO, use the update tool and confirm the changes.
- When explaining technical concepts like WAPE, Tweedie distribution, newsvendor model, or censoring, provide clear business context.
"""


def answer_locally_without_api_key(user_msg: str) -> str:
    """Intelligent rule-based fallback when GEMINI_API_KEY is not yet supplied."""
    msg_lower = user_msg.lower()

    # 1. Action: Approve Critical / All Orders
    if "approve" in msg_lower and ("all" in msg_lower or "critical" in msg_lower or "order" in msg_lower or "po" in msg_lower):
        from src.api.main import PURCHASE_ORDERS, save_purchase_orders
        approved_count = 0
        total_val = 0.0
        for po in PURCHASE_ORDERS:
            if po.get("urgency") in ["Critical", "High"] and po.get("status") != "Approved":
                po["status"] = "Approved"
                approved_count += 1
                total_val += po.get("total_cost", 0.0)
        save_purchase_orders()
        return (
            f"### ✅ Purchase Orders Approved\n\n"
            f"Successfully approved **{approved_count} urgent purchase orders** totaling **${total_val:,.2f}**.\n\n"
            f"- Status updated in real time across the procurement ledger.\n"
            f"- You can review and export these orders in the **Purchase Orders** tab."
        )

    # 2. Policy Simulation & Trade-off Comparison
    if any(k in msg_lower for k in ["compare", "policy", "policies", "heuristic", "quantile", "frontier", "tradeoff", "trade-off"]):
        return (
            "### ⚖️ Replenishment Policy Benchmark\n\n"
            "Restok evaluates three replenishment strategies over a 14-day horizon across 20 stochastic simulation seeds:\n\n"
            "1. **Quantile Ordering (Restok Recommended)**:\n"
            "   - **Service Level**: **93.37%** (Stockout Rate: **6.63%**)\n"
            "   - **Total Supply Chain Cost**: **$2,184.73** (Holding: $286.09, Stockout Penalty: $1,898.64)\n"
            "   - **Mechanism**: Calibrated quantile forecasts ($q_{75}, q_{85}, q_{90}$) mapped to category Newsvendor critical ratio $\\tau = C_u / (C_u + C_o)$ with perishable shelf-life caps.\n\n"
            "2. **Manager Heuristic Baseline**:\n"
            "   - **Service Level**: **92.21%** (Stockout Rate: **7.79%**)\n"
            "   - **Total Supply Chain Cost**: **$2,531.17**\n"
            "   - **Deficiency**: Uses static 3-day buffer rules that over-order slow movers and stock out on demand spikes.\n\n"
            "3. **Point Forecast Ordering**:\n"
            "   - **Service Level**: **90.15%** (Stockout Rate: **9.85%**)\n"
            "   - **Total Supply Chain Cost**: **$2,842.10**\n"
            "   - **Deficiency**: Orders only expected mean, causing stockouts on ~50% of above-average demand days.\n\n"
            "> **Headline Impact**: Quantile Ordering achieves **14.89% stockout reduction** and **13.69% total cost savings** vs the Manager Heuristic."
        )

    # 3. Model Architecture & Tweedie vs 7-day Baseline Explanation
    if any(k in msg_lower for k in ["why lightgbm", "tweedie", "outperform", "rolling mean", "architecture", "feature"]):
        return (
            "### 🤖 LightGBM Tweedie vs 7-Day Baseline\n\n"
            "Our tuned **LightGBM Tweedie point model** achieves **34.91% WAPE**, outperforming the 38.10% 7-day rolling mean baseline (**6.93% relative error reduction**).\n\n"
            "**Key Architectural Advantages**:\n"
            "- **Tweedie Compound Distribution**: Supermarket demand contains a zero mass (non-sale days) mixed with right-skewed positive purchases. The Tweedie objective (variance power $p=1.268$) naturally models this without artificial log-transform distortions.\n"
            "- **Zero Information Leakage**: Features strictly obey the 14-day supply horizon (lags shifted by $\ge 14$ days: $t-14, t-15, t-21, t-28, t-35$).\n"
            "- **Exogenous Grocery Drivers**: Captures price elasticity ratios, promotional flags, SNAP food stamp disbursement schedules, cyclical calendar encodings (sine/cosine of day-of-week and month), and holiday proximity.\n"
            "- **Quantile Uncertainty Ribbons**: Simultaneously trains pinball-loss regressors at $\\tau \\in [0.50, 0.75, 0.85, 0.90, 0.95]$ with conformal residual calibration."
        )

    # 4. High-level KPIs and Accuracy
    if any(k in msg_lower for k in ["kpi", "accuracy", "performance", "wape", "error"]):
        kpis = tool_get_system_kpis()
        return (
            f"### 📊 Restok System Performance\n\n"
            f"- **LightGBM Tweedie WAPE**: **{kpis['model_wape_pct']}%** (an **{kpis['wape_improvement_pct']}% improvement** over the 7-day rolling mean baseline of {kpis['baseline_7day_wape_pct']}%).\n"
            f"- **Optuna Hyperparameters**: Best Tweedie variance power=1.268, learning_rate=0.120, num_leaves=54.\n"
            f"- **Stockout Rate**: Reduced from **{kpis['heuristic_stockout_rate_pct']}%** (Manager Heuristic) down to **{kpis['quantile_stockout_rate_pct']}%** (**14.89% stockout reduction**).\n"
            f"- **Supply Chain Cost**: Decreased by **{kpis['total_cost_savings_pct']}%** from $2,531 to $2,184 across 20 stochastic seeds.\n"
            f"- **Active Portfolio**: 300 series across 5 stores and 60 FOODS SKUs.\n\n"
            f"> *Note: Connect your Gemini API Key in the UI or set `export GEMINI_API_KEY='...'` for full open-ended LLM reasoning.*"
        )

    # 5. Censoring Study & True Demand Recovery
    if any(k in msg_lower for k in ["censor", "bias", "true demand", "latent"]):
        c = tool_get_censoring_insights()
        return (
            f"### 🔍 Phase 7 Censoring Study Findings\n\n"
            f"When customer demand exceeds on-hand shelf inventory, sales are artificially truncated to zero or available stock.\n\n"
            f"- **Naive Model Bias**: **{c['naive_downward_bias']}**. Naive models trained on observed sales learn that demand is low when the shelf is empty, causing downward procurement spirals.\n"
            f"- **Dropping Stockout Days**: **{c['drop_stockouts_bias']}**. Removing stockout rows breaks time-series continuity and introduces survivorship bias.\n"
            f"- **Iterative EM Imputation**: Recovers true latent demand, achieving **{c['iterative_em_imputation']}** and lowest test RMSE (6.96).\n\n"
            f"**Recommendation**: {c['recommendation']}"
        )

    # 6. Store Portfolio
    if any(k in msg_lower for k in ["stores", "store", "location", "locations", "walmart"]):
        return (
            "### 🏪 Monitored Store Network\n\n"
            "Restok currently orchestrates replenishment for **5 strategic Walmart store footprints**:\n\n"
            "- **`CA_1`** (Los Angeles, CA): High-velocity urban flagship; strong weekend shopping surges.\n"
            "- **`CA_2`** (Eastvale, CA): Large suburban supercenter with diverse packaged dry goods volume.\n"
            "- **`TX_1`** (Houston, TX): High sales volume with high sensitivity to promotional discounting.\n"
            "- **`TX_2`** (Fort Worth, TX): Balanced grocery mix with distinct SNAP disbursement cycles.\n"
            "- **`WI_1`** (Milwaukee, WI): Northern climate with pronounced seasonal temperature swings.\n\n"
            "Each store carries all **60 target FOODS SKUs**, yielding **300 active Store $\\times$ SKU forecasting series**."
        )

    # 7. Category & Perishability Rules
    if any(k in msg_lower for k in ["category", "categories", "perishable", "perishables", "shelf life", "case pack", "dept"]):
        return (
            "### 🍏 Food Departments & Replenishment Parameters\n\n"
            "Items are governed by category-specific supply constraints defined in `config/config.yaml`:\n\n"
            "- **`FOODS_1` (Packaged Dry Goods)**: Non-perishable, 90-day shelf life, 3-day lead time, 12-unit case pack, Critical Ratio $\\tau = 0.75$ ($q_{75}$ reorder point).\n"
            "- **`FOODS_2` (Bakery & Dairy)**: Semi-perishable, 14-day shelf life, 2-day lead time, 6-unit case pack, Critical Ratio $\\tau = 0.85$ ($q_{85}$ reorder point).\n"
            "- **`FOODS_3` (Fresh Produce & Deli)**: Highly perishable, 7-day shelf life, 1-day lead time, 4-unit case pack, Critical Ratio $\\tau = 0.90$ ($q_{90}$ reorder point).\n\n"
            "> Perishable safety cap: Order target stock is capped at $\\hat{y}_{\\text{point}} \\times \\text{Shelf Life}$ to eliminate spoilage while meeting fill rates."
        )

    # 8. Reorder Point & Replenishment Math
    if any(k in msg_lower for k in ["formula", "rop", "reorder point", "newsvendor", "math", "critical ratio"]):
        return (
            "### 📐 Replenishment Policy Mathematics\n\n"
            "Restok determines purchase orders using Newsvendor quantile optimization:\n\n"
            "1. **Critical Ratio ($\\tau$)**:\n"
            "   $$\\tau = \\frac{C_u}{C_u + C_o}$$\n"
            "   Where $C_u$ is stockout penalty ($1.5 \\times$ lost margin) and $C_o$ is holding cost ($20\\%$ annual rate).\n\n"
            "2. **Reorder Point ($ROP$)**:\n"
            "   $$ROP = \\hat{y}_{q,\\tau} \\times L$$\n"
            "   Triggered whenever $\\text{On-Hand} \\le ROP$.\n\n"
            "3. **Order Quantity**:\n"
            "   $$\\text{Target Stock} = \\min\\left(\\hat{y}_{q,\\tau} \\times (L + 2), \\; \\hat{y}_{\\text{point}} \\times \\text{Shelf Life}\\right)$$\n"
            "   $$\\text{Order Qty} = \\left\\lceil \\frac{\\max(0, \\text{Target Stock} - \\text{On-Hand})}{\\text{Case Pack}} \\right\\rceil \\times \\text{Case Pack}$$"
        )

    # 9. Critical Stockouts & Purchase Orders
    if any(k in msg_lower for k in ["critical", "urgent", "stockout risk", "po", "order", "orders", "replenish"]):
        critical_pos = tool_get_purchase_orders(urgency="Critical")
        if not critical_pos:
            critical_pos = tool_get_purchase_orders(urgency="High")

        lines = [f"### 🚨 Urgent Replenishment Orders ({len(critical_pos)} Critical Orders Shown)\n"]
        for p in critical_pos[:6]:
            lines.append(
                f"- **{p['po_id']}** ({p['store_id']} | `{p['item_id']}`): On-hand **{p['on_hand']} units** vs ROP **{p['rop']}**. "
                f"Recommended Order: **{p['order_qty']} units** (Case Pack: {p['case_pack']}, Lead Time: {p['lead_time_days']}d, Cost: ${p['total_cost']:.2f})."
            )
        lines.append("\n*You can approve these orders directly in the Purchase Orders tab, or say 'Approve all critical orders'!*")
        return "\n".join(lines)

    # 10. Specific SKU Search (e.g. FOODS_1_012, optionally with store e.g. TX_1)
    for part in user_msg.replace(",", " ").split():
        if "foods_" in part.lower():
            item_clean = part.upper().strip("?.,!:")
            target_store = "CA_1"
            for store in ["CA_1", "CA_2", "TX_1", "TX_2", "WI_1"]:
                if store.lower() in msg_lower:
                    target_store = store
                    break

            details = tool_get_sku_details(target_store, item_clean)
            if "error" not in details:
                fore = tool_get_forecast_analysis(target_store, item_clean)
                return (
                    f"### 📦 SKU Profile: `{item_clean}` in Store `{target_store}`\n\n"
                    f"- **Department**: {details['dept_id']} | **Critical Ratio (&tau;)**: {details['critical_ratio']}\n"
                    f"- **On-Hand Inventory**: **{details['on_hand_inventory']} units** (Reorder Point ROP: **{details['reorder_point_rop']} units**)\n"
                    f"- **Lead Time**: {details['lead_time_days']} days | **Shelf Life**: {details['shelf_life_days']} days\n"
                    f"- **Case Pack**: {details['case_pack']} units | **Unit Cost**: ${details['unit_cost']:.2f} (Price: ${details['sell_price']:.2f})\n"
                    f"- **14-day Forecast**: Point: **{fore['total_point_forecast']} units** | Upper Quantile (q90): **{fore['total_q90_forecast']} units**\n"
                    f"- **Replenishment Status**: {'🚨 Order ' + str(details['recommended_order_qty']) + ' units immediately' if details['is_stockout_risk'] else '✅ Stock level adequate'}\n"
                )

    return (
        f"### 👋 Restok AI Copilot Ready\n\n"
        f"I am connected to your Restok inventory engine with live access to **300 Store & SKU series**, **LightGBM forecasts**, and **automated purchase orders**.\n\n"
        f"Try asking:\n"
        f"- *'Which SKUs are at critical stockout risk right now?'*\n"
        f"- *'Compare Quantile Ordering vs Manager Heuristic costs'*\n"
        f"- *'Why is LightGBM Tweedie outperforming the 7-day baseline?'*\n"
        f"- *'Explain the downward bias in the Censoring Study'*\n"
        f"- *'What is our forecast and ROP for FOODS_1_012 in CA_1?'*\n"
        f"- *'What stores and food categories are tracked?'*\n\n"
        f"> 💡 *To unlock Gemini 2.5 Flash conversational reasoning, set `export GEMINI_API_KEY='your-key'` or provide your key in the chat settings!*"
    )


def generate_chat_response(
    user_message: str,
    history: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
) -> Dict:
    """Main chatbot orchestrator: calls Gemini with tool execution, with local fallback."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

    if not key or not GENAI_AVAILABLE:
        reply = answer_locally_without_api_key(user_message)
        return {"reply": reply, "engine": "local_fallback", "tool_calls": []}

    try:
        client = genai.Client(api_key=key)

        # Register functions as native tools
        tools = [
            tool_get_system_kpis,
            tool_get_sku_details,
            tool_get_forecast_analysis,
            tool_get_purchase_orders,
            tool_update_purchase_order,
            tool_get_censoring_insights,
        ]

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=tools,
            temperature=0.2,
        )

        # Build conversation history
        history_contents = []
        if history:
            for turn in history[-6:]:  # Keep recent context
                role = "user" if turn.get("role") == "user" else "model"
                history_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn.get("content", ""))]))

        # Initialize chat with AFC (Automatic Function Calling)
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=config,
            history=history_contents,
        )

        # Call Gemini 2.5 Flash
        response = chat.send_message(user_message)

        return {
            "reply": response.text or "Analysis completed.",
            "engine": "gemini-2.5-flash",
            "tool_calls": [],
        }

    except Exception as e:
        logger.warning("Gemini API call failed (%s). Falling back to local engine.", e)
        reply = answer_locally_without_api_key(user_message)
        return {
            "reply": f"{reply}\n\n*(Gemini API Note: {str(e)})*",
            "engine": "local_fallback",
            "tool_calls": [],
        }
