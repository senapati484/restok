"""Restok — Streamlit workspace for demand forecasting and replenishment.

Run it with:
    streamlit run streamlit_app.py

Sections:
1. Overview     - KPIs, urgent SKUs, forecast chart, project benchmarks
2. Data         - demo or your own CSV, validation and preview
3. Forecast     - per-SKU forecast with uncertainty band + downloads
4. Orders       - editable purchase-order plan with batch actions
5. Insights     - model, policy, trade-off and censoring study results
6. Copilot      - grounded AI assistant (Gemini with local fallback)
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.app.data import (
    TEMPLATE_COLUMNS,
    DataReport,
    clean_sales_data,
    infer_defaults,
    load_csv_bytes,
    load_demo_data,
    template_csv_bytes,
)
from src.app.forecast import AnalysisResult, Settings, run_analysis
from src.api.chatbot import generate_chat_response
from src.config import PROCESSED_DIR

st.set_page_config(
    page_title="Restok",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 1.6rem; max-width: 1180px; }
h1, h2, h3 { color: #0f172a; }
[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { font-size: 1.55rem; color: #0f172a; }
[data-testid="stMetricLabel"] { color: #475569; }
.hero {
    background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #6366f1;
    padding: 1.15rem 1.4rem; border-radius: 10px; margin-bottom: 1rem;
}
.hero h1 { color: #0f172a; font-size: 1.5rem; margin: 0 0 0.25rem 0; }
.hero p { color: #475569; margin: 0; font-size: 0.95rem; }
.hint { color: #64748b; font-size: 0.84rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

POLICY_LABELS = {
    "manager_heuristic": "Manager heuristic",
    "point_forecast": "Point forecast",
    "quantile_ordering": "Quantile ordering",
}
URGENCY_RANK = {"Critical": 0, "High": 1, "Normal": 2, "Covered": 3}


# --------------------------------------------------------------------------- #
# Data / caching
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _load_demo() -> tuple[pd.DataFrame, DataReport]:
    return clean_sales_data(load_demo_data(max_series=120))


@st.cache_data(show_spinner=False)
def _load_project_data() -> dict[str, pd.DataFrame]:
    """Loads the bundled M5 study results used by the Insights and Copilot tabs."""
    files = {
        "metrics": "metrics.csv",
        "policy": "policy_kpi_summary.csv",
        "tradeoff": "policy_tradeoff_curve.csv",
        "censoring": "censoring_comparison.csv",
    }
    output: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = PROCESSED_DIR / filename
        output[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return output


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> alt.LayerChart:
    x = alt.X("date:T", title=None)
    band = (
        alt.Chart(forecast)
        .mark_area(color="#6366f1", opacity=0.15)
        .encode(x=x, y=alt.Y("lower:Q", title="Units"), y2="upper:Q")
    )
    forecast_line = (
        alt.Chart(forecast)
        .mark_line(color="#6366f1", strokeWidth=2.2, strokeDash=[6, 4])
        .encode(x=x, y="forecast:Q", tooltip=["date:T", "forecast:Q", "lower:Q", "upper:Q"])
    )
    history_line = (
        alt.Chart(history)
        .mark_line(color="#334155", strokeWidth=2)
        .encode(x=x, y="units_sold:Q", tooltip=["date:T", "units_sold:Q"])
    )
    divider = (
        alt.Chart(pd.DataFrame({"date": [forecast["date"].min()]}))
        .mark_rule(color="#cbd5e1", strokeDash=[3, 3])
        .encode(x="date:T")
    )
    return (
        (band + forecast_line + history_line + divider)
        .properties(height=330)
        .configure_axis(grid=False, labelColor="#475569", titleColor="#475569")
        .configure_view(stroke=None)
        .interactive()
    )


def _wape_chart(metrics: pd.DataFrame) -> alt.LayerChart:
    frame = metrics.dropna(subset=["WAPE"]).copy()
    frame["WAPE (%)"] = frame["WAPE"] * 100
    frame["Highlight"] = frame["Model"].str.contains("Tweedie", na=False)
    base = alt.Chart(frame).mark_bar().encode(
        x=alt.X("Model:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-25)),
        y=alt.Y("WAPE (%):Q", title="Forecast error (WAPE %)"),
        color=alt.condition(
            alt.datum.Highlight, alt.value("#4f46e5"), alt.value("#c7d2fe")
        ),
        tooltip=["Model:N", alt.Tooltip("WAPE (%):Q", format=".1f")],
    )
    labels = base.mark_text(dy=-6, color="#475569").encode(
        text=alt.Text("WAPE (%):Q", format=".1f")
    )
    return (base + labels).properties(height=300)


def _policy_chart(policy: pd.DataFrame, column: str, title: str) -> alt.Chart:
    frame = policy.copy()
    frame["Policy"] = frame["Policy"].map(POLICY_LABELS).fillna(frame["Policy"])
    return (
        alt.Chart(frame)
        .mark_bar(color="#818cf8")
        .encode(
            x=alt.X("Policy:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{column}:Q", title=title),
            tooltip=["Policy:N", alt.Tooltip(f"{column}:Q", format=".2f")],
        )
        .properties(height=260)
    )


def _tradeoff_chart(tradeoff: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(tradeoff)
        .mark_line(point=True, color="#4f46e5", strokeWidth=2.2)
        .encode(
            x=alt.X("service_level_pct:Q", title="In-stock level (%)"),
            y=alt.Y("total_cost_usd:Q", title="Total cost (USD)"),
            tooltip=[
                alt.Tooltip("service_level_pct:Q", title="In-stock %", format=".2f"),
                alt.Tooltip("stockout_rate_pct:Q", title="Stockout %", format=".2f"),
                alt.Tooltip("total_cost_usd:Q", title="Total cost", format=".2f"),
            ],
        )
        .properties(height=300)
        .interactive()
    )


def _censoring_chart(censoring: pd.DataFrame, column: str, title: str) -> alt.Chart:
    frame = censoring.dropna(subset=[column]).copy()
    frame[title] = frame[column] * 100
    return (
        alt.Chart(frame)
        .mark_bar(color="#a5b4fc")
        .encode(
            x=alt.X("Method:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(f"{title}:Q"),
            tooltip=["Method:N", alt.Tooltip(f"{title}:Q", format=".2f")],
        )
        .properties(height=260, title=title)
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def open_browser_hint() -> None:
    st.sidebar.caption("Tip: run `./run.sh` to open this app automatically.")


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def _top_urgent(table: pd.DataFrame) -> pd.DataFrame:
    action = table[table["recommended_qty"] > 0].copy()
    if action.empty:
        return action
    action["_rank"] = action["urgency"].map(URGENCY_RANK).fillna(4)
    return action.sort_values(["_rank", "order_cost"], ascending=[True, False])


def _project_kpis() -> None:
    data = _load_project_data()
    metrics, policy = data["metrics"], data["policy"]
    if metrics.empty or policy.empty:
        return

    tweedie = metrics[metrics["Model"].str.contains("Tweedie", na=False)]
    baseline = metrics[metrics["Model"].str.contains("7-day", na=False)]
    heuristic = policy[policy["Policy"] == "manager_heuristic"]
    quantile = policy[policy["Policy"] == "quantile_ordering"]
    if tweedie.empty or baseline.empty or heuristic.empty or quantile.empty:
        return

    model_wape = float(tweedie["WAPE"].iloc[0]) * 100
    base_wape = float(baseline["WAPE"].iloc[0]) * 100
    improvement = (base_wape - model_wape) / base_wape * 100
    heur_stock = float(heuristic["stockout_rate_pct_mean"].iloc[0])
    quant_stock = float(quantile["stockout_rate_pct_mean"].iloc[0])
    stock_reduction = (heur_stock - quant_stock) / heur_stock * 100
    heur_cost = float(heuristic["total_cost_usd_mean"].iloc[0])
    quant_cost = float(quantile["total_cost_usd_mean"].iloc[0])
    cost_reduction = (heur_cost - quant_cost) / heur_cost * 100

    st.markdown("#### Project benchmarks (M5 study)")
    st.caption(
        "These come from the full 300-series M5 study bundled with the project, "
        "independent of the data selected above."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model WAPE", f"{model_wape:.1f}%", f"{-improvement:.1f}% vs baseline")
    c2.metric("Stockout reduction", f"{stock_reduction:.1f}%", "quantile vs heuristic")
    c3.metric("Cost reduction", f"{cost_reduction:.1f}%", "quantile vs heuristic")
    c4.metric("Series studied", "300", "5 stores x 60 SKUs")


def _overview_tab(result: AnalysisResult | None, report: DataReport | None) -> None:
    st.subheader("Overview")

    if result is None:
        st.info("Run an analysis in the sidebar to see your forecast and orders.")
        _project_kpis()
        return

    table = result.replenishment
    actionable = table[table["recommended_qty"] > 0]
    urgent = _top_urgent(table)
    spend = float(actionable["order_cost"].sum()) if not actionable.empty else 0.0
    wape = result.metrics["wape"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Forecast error",
        "n/a" if wape is None else f"{wape * 100:.0f}%",
        help="WAPE on held-out days. Lower is better.",
    )
    c2.metric("Orders to place", f"{len(actionable):,}")
    c3.metric("Critical items", f"{int((table['urgency'] == 'Critical').sum()):,}")
    c4.metric("Estimated spend", f"${spend:,.0f}")

    if report is not None:
        st.caption(
            f"{report.series:,} SKUs · {report.rows_out:,} rows · "
            f"{(report.date_max - report.date_min).days + 1:,} days of history"
        )

    if urgent.empty:
        st.success("Every SKU is above its reorder point. No orders needed.")
        _project_kpis()
        return

    left, right = st.columns([3, 2])
    focus = urgent.iloc[0]
    focus_fc = result.forecasts[
        (result.forecasts["store_id"] == focus["store_id"])
        & (result.forecasts["item_id"] == focus["item_id"])
    ].sort_values("date")
    focus_hist = result.history[
        (result.history["store_id"] == focus["store_id"])
        & (result.history["item_id"] == focus["item_id"])
    ].sort_values("date").tail(60)

    with left:
        st.markdown(f"#### Highest-priority SKU · {focus['store_id']} / {focus['item_id']}")
        st.altair_chart(_forecast_chart(focus_hist, focus_fc), width="stretch")

    with right:
        st.markdown("#### Needs attention now")
        top = urgent.head(6)[
            ["store_id", "item_id", "recommended_qty", "urgency", "order_cost"]
        ].copy()
        top.columns = ["Store", "Item", "Order qty", "Priority", "Cost"]
        st.dataframe(top, width="stretch", hide_index=True)

        if table["on_hand_assumed"].any():
            st.caption(
                "Stock on hand was assumed. Add an `on_hand` column for exact quantities."
            )

    _project_kpis()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _csv_help() -> None:
    with st.expander("CSV format"):
        st.markdown(
            "One row per **store, item and day**. Only `date`, `item_id` and "
            "`units_sold` are required. Aliases such as `ds`, `sku` and `sales` "
            "are detected automatically."
        )
        st.code(", ".join(TEMPLATE_COLUMNS), language="text")
        st.markdown(
            "Optional columns: `sell_price`, `unit_cost`, `on_hand`, `lead_time`, "
            "`case_pack`, `is_promotion`, `shelf_life`."
        )


def _data_tab(clean_df: pd.DataFrame | None, report: DataReport | None) -> None:
    if clean_df is None or report is None:
        st.info("Choose the demo or upload a CSV in the sidebar to begin.")
        _csv_help()
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs tracked", f"{report.series:,}")
    c2.metric("Days of history", f"{(report.date_max - report.date_min).days + 1:,}")
    c3.metric("Rows loaded", f"{report.rows_out:,}")
    c4.metric("Merged duplicates", f"{report.duplicates:,}")

    st.caption(f"{report.date_min:%d %b %Y} to {report.date_max:%d %b %Y}")

    for warning in report.warnings:
        st.warning(warning)

    missing = [c for c in report.missing_optional if c not in ("store_id", "is_promotion")]
    if missing:
        st.info(
            "Using default values for: " + ", ".join(missing)
            + ". Add these columns for more precise orders."
        )

    cols = st.columns(2)
    with cols[0]:
        with st.expander("Preview data"):
            st.dataframe(clean_df.head(100), width="stretch", hide_index=True)
    with cols[1]:
        with st.expander("Recognised columns"):
            if report.mapped_columns:
                mapping = pd.DataFrame(
                    [{"Your column": s, "Used as": t} for t, s in report.mapped_columns.items()]
                )
                st.dataframe(mapping, hide_index=True, width="stretch")

    st.download_button(
        "Download CSV template",
        data=template_csv_bytes(),
        file_name="restok_template.csv",
        mime="text/csv",
    )
    _csv_help()


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
def _forecast_tab(result: AnalysisResult) -> None:
    metrics = result.metrics
    wape = metrics["wape"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Forecast error",
        "n/a" if wape is None else f"{wape * 100:.0f}%",
        help="WAPE on the held-out last days. Lower is better.",
    )
    c2.metric(
        "Expected demand",
        f"{result.forecasts['forecast'].sum():,.0f}",
        help=f"Total units over the next {metrics['horizon']} days.",
    )
    c3.metric("Average per day", f"{result.forecasts['forecast'].mean():.1f}")

    if wape is None:
        st.info(
            "Not enough history for an accuracy check, so a simple weekly average "
            "is being used. Add more days of data for the machine-learning forecast."
        )
    elif wape > 0.6:
        st.warning(
            "Forecast error is high. Treat quantities as directional and consider "
            "adding more history, price or promotion data."
        )

    stores = sorted(result.forecasts["store_id"].unique())
    store = st.selectbox("Store", stores)
    items = sorted(
        result.forecasts[result.forecasts["store_id"] == store]["item_id"].unique()
    )
    item = st.selectbox("Item", items)

    series_fc = result.forecasts[
        (result.forecasts["store_id"] == store) & (result.forecasts["item_id"] == item)
    ].sort_values("date")
    series_hist = result.history[
        (result.history["store_id"] == store) & (result.history["item_id"] == item)
    ].sort_values("date").tail(60)

    st.altair_chart(_forecast_chart(series_hist, series_fc), width="stretch")

    with st.expander("Daily forecast and download"):
        display = series_fc[["date", "forecast", "lower", "upper"]].copy()
        display.columns = ["Date", "Forecast", "Low", "High"]
        display[["Forecast", "Low", "High"]] = display[["Forecast", "Low", "High"]].round(2)
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button(
            "Download this forecast",
            data=series_fc.to_csv(index=False),
            file_name=f"forecast_{store}_{item}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download all forecasts",
            data=result.forecasts.to_csv(index=False),
            file_name="restok_forecasts.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
def _orders_tab(result: AnalysisResult) -> None:
    table = result.replenishment.copy()
    if table.empty:
        st.info("No replenishment rows were produced.")
        return

    if "editor_version" not in st.session_state:
        st.session_state["editor_version"] = 0

    actionable = table[table["recommended_qty"] > 0]
    spend = float(actionable["order_cost"].sum()) if not actionable.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Orders to place", f"{len(actionable):,}")
    c2.metric("Critical items", f"{int((table['urgency'] == 'Critical').sum()):,}")
    c3.metric("Estimated cost", f"${spend:,.0f}")
    c4.metric("Covered SKUs", f"{int((table['urgency'] == 'Covered').sum()):,}")

    if table["on_hand_assumed"].any():
        st.info(
            "Stock on hand was assumed because your file has no `on_hand` column. "
            "Add it for precise orders."
        )

    show_all = st.toggle("Show SKUs that do not need an order", value=False)
    view = table if show_all else actionable
    if view.empty:
        st.success("Every SKU is above its reorder point. No orders needed.")
        return

    display = view[
        [
            "store_id", "item_id", "on_hand", "reorder_point", "recommended_qty",
            "lead_time", "urgency", "status",
        ]
    ].copy()

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Approve all critical", width="stretch"):
        st.session_state["bulk_action"] = "critical"
    if b2.button("Clear edits", width="stretch"):
        st.session_state["bulk_action"] = "reset"

    action = st.session_state.pop("bulk_action", None)
    if action == "critical":
        display.loc[display["urgency"] == "Critical", "status"] = "Approved"
        st.session_state["editor_version"] += 1
    elif action == "reset":
        st.session_state["editor_version"] += 1

    edited = st.data_editor(
        display,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        key=f"orders_editor_{st.session_state['editor_version']}",
        column_config={
            "store_id": st.column_config.TextColumn("Store", disabled=True),
            "item_id": st.column_config.TextColumn("Item", disabled=True),
            "on_hand": st.column_config.NumberColumn("Stock", disabled=True, format="%.1f"),
            "reorder_point": st.column_config.NumberColumn("Reorder at", disabled=True, format="%.1f"),
            "recommended_qty": st.column_config.NumberColumn("Order qty", min_value=0, step=1),
            "lead_time": st.column_config.NumberColumn("Lead days", disabled=True),
            "urgency": st.column_config.TextColumn("Priority", disabled=True),
            "status": st.column_config.SelectboxColumn(
                "Status", options=["Pending", "Approved", "Ignored"], required=True
            ),
        },
    )

    unit_cost = view["unit_cost"].to_numpy()
    edited["unit_cost"] = unit_cost
    edited["order_cost"] = (edited["recommended_qty"].to_numpy() * unit_cost).round(2)

    e1, e2 = st.columns(2)
    e1.metric("Plan value (after edits)", f"${edited['order_cost'].sum():,.0f}")
    e2.metric("Approved rows", f"{int((edited['status'] == 'Approved').sum()):,}")

    st.download_button(
        "Download order plan",
        data=edited.to_csv(index=False),
        file_name="restok_orders.csv",
        mime="text/csv",
    )

    with st.expander("How quantities are calculated"):
        st.markdown(
            "**Reorder point** = average daily demand x lead time + safety stock\n\n"
            "**Safety stock** = z x demand variability x square root of lead time, "
            f"where z comes from your {result.metrics['service_level']:.0%} target.\n\n"
            "**Order quantity** = target stock - stock on hand, rounded up to a full case pack."
        )


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
def _insights_tab() -> None:
    data = _load_project_data()
    metrics, policy = data["metrics"], data["policy"]
    tradeoff, censoring = data["tradeoff"], data["censoring"]

    st.subheader("Project insights")
    st.caption(
        "Results from the full 300-series M5 study bundled with the project. "
        "They are independent of the data selected in the sidebar."
    )

    if metrics.empty:
        st.info("Study results are not available. Run `./run.sh pipeline` to generate them.")
        return

    _project_kpis()
    st.markdown("---")

    st.markdown("#### Forecasting model comparison")
    st.altair_chart(_wape_chart(metrics), width="stretch")
    with st.expander("Raw model metrics"):
        st.dataframe(metrics, width="stretch", hide_index=True)

    if not policy.empty:
        st.markdown("#### Replenishment policy comparison")
        left, right = st.columns(2)
        with left:
            st.altair_chart(
                _policy_chart(policy, "stockout_rate_pct_mean", "Stockout rate (%)"),
                width="stretch",
            )
        with right:
            st.altair_chart(
                _policy_chart(policy, "total_cost_usd_mean", "Total cost (USD)"),
                width="stretch",
            )
        with st.expander("Raw policy metrics (95% confidence intervals)"):
            st.dataframe(policy, width="stretch", hide_index=True)

    if not tradeoff.empty:
        st.markdown("#### Service level vs total cost")
        st.caption(
            "Each point is a different safety-stock policy. Moving right improves "
            "availability; moving up increases total cost."
        )
        st.altair_chart(_tradeoff_chart(tradeoff), width="stretch")

    if not censoring.empty:
        st.markdown("#### Censoring study")
        st.caption(
            "When stock runs out, real demand is hidden. Training on censored sales "
            "under-forecasts demand, as shown by the negative bias."
        )
        left, right = st.columns(2)
        with left:
            st.altair_chart(
                _censoring_chart(censoring, "WAPE_vs_True_Demand", "WAPE vs true demand (%)"),
                width="stretch",
            )
        with right:
            st.altair_chart(
                _censoring_chart(censoring, "Forecast_Bias", "Forecast bias (%)"),
                width="stretch",
            )
        with st.expander("Raw censoring results"):
            st.dataframe(censoring, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Copilot
# --------------------------------------------------------------------------- #
def _copilot_send(message: str, api_key: str | None) -> None:
    messages = st.session_state["copilot"]
    messages.append({"role": "user", "content": message})
    with st.spinner("Thinking..."):
        response = generate_chat_response(message, history=messages, api_key=api_key)
    messages.append(
        {
            "role": "assistant",
            "content": response["reply"],
            "engine": response.get("engine", "local_fallback"),
        }
    )


def _copilot_tab() -> None:
    st.subheader("Restok Copilot")
    st.caption(
        "Grounded assistant for the bundled M5 project: forecasts, purchase orders, "
        "policy trade-offs and the censoring study."
    )

    with st.expander("Gemini API key (optional)"):
        api_key = st.text_input(
            "API key",
            type="password",
            value=st.session_state.get("gemini_key", ""),
            help="Leave blank to use the built-in local analytical engine.",
        )
        if api_key:
            st.session_state["gemini_key"] = api_key

    if "copilot" not in st.session_state:
        st.session_state["copilot"] = []

    for message in st.session_state["copilot"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("engine"):
                st.caption(f"Engine: {message['engine']}")

    if not st.session_state["copilot"]:
        examples = [
            "What is our model WAPE and how does it compare to the baseline?",
            "Compare quantile ordering with the manager heuristic.",
            "Which purchase orders are critical right now?",
            "Explain the censoring study and its bias.",
        ]
        cols = st.columns(2)
        for index, example in enumerate(examples):
            if cols[index % 2].button(example, width="stretch"):
                _copilot_send(example, st.session_state.get("gemini_key"))
                st.rerun()

    prompt = st.chat_input("Ask about demand, orders, policies or stockouts...")
    if prompt:
        _copilot_send(prompt, st.session_state.get("gemini_key"))
        st.rerun()

    if st.session_state["copilot"]:
        if st.button("Clear conversation"):
            st.session_state["copilot"] = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.markdown(
        """
<div class="hero">
  <h1>Restok</h1>
  <p>Demand forecasting, replenishment planning and supply-chain insights
  from your own sales data.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Data")
    source = st.sidebar.radio(
        "Data source",
        ["Use demo data", "Upload my CSV"],
        label_visibility="collapsed",
    )
    demo_mode = source.startswith("Use demo")

    uploaded = None
    if not demo_mode:
        uploaded = st.sidebar.file_uploader("CSV file", type=["csv"], key="uploader")
        st.sidebar.download_button(
            "Download CSV template",
            data=template_csv_bytes(),
            file_name="restok_template.csv",
            mime="text/csv",
            width="stretch",
        )

    clean_df: pd.DataFrame | None = None
    report: DataReport | None = None

    if demo_mode:
        try:
            clean_df, report = _load_demo()
            if st.session_state.get("result_source") != "demo":
                st.session_state["auto_run"] = True
        except Exception as exc:
            st.error(str(exc))
    elif uploaded is not None:
        try:
            clean_df, report = clean_sales_data(load_csv_bytes(uploaded.getvalue()))
            signature = (uploaded.name, uploaded.size)
            if st.session_state.get("upload_signature") != signature:
                st.session_state["upload_signature"] = signature
                st.session_state["auto_run"] = True
        except ValueError as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

    defaults = infer_defaults(clean_df) if clean_df is not None else infer_defaults(pd.DataFrame())
    st.sidebar.markdown("### Settings")
    horizon = st.sidebar.slider("Forecast days", 7, 42, 14)
    service_pct = st.sidebar.slider("Target in-stock level (%)", 50, 99, 90)
    lead_time = st.sidebar.number_input(
        "Lead time (days)", min_value=1, max_value=60,
        value=max(1, int(round(defaults["lead_time"]))),
    )
    with st.sidebar.expander("Advanced settings"):
        case_pack = st.number_input(
            "Default case pack", min_value=1, max_value=500,
            value=max(1, int(round(defaults["case_pack"]))),
        )
        review_period = st.slider("Days between orders", 1, 30, 7)
        cover_days = st.number_input(
            "Assumed stock on hand (days)", min_value=0.0, max_value=90.0,
            value=10.0, step=1.0,
            help="Used only when your file has no on-hand inventory column.",
        )

    st.sidebar.caption("Values in your file override these defaults.")
    run_requested = st.sidebar.button("Run analysis", type="primary", width="stretch")

    settings = Settings(
        horizon=int(horizon),
        lead_time=int(lead_time),
        case_pack=int(case_pack),
        review_period=int(review_period),
        service_level=service_pct / 100.0,
        cover_days=float(cover_days),
        sell_price=float(defaults["sell_price"]),
        unit_cost=float(defaults["unit_cost"]),
        shelf_life=int(round(defaults["shelf_life"])),
    )

    auto_run = st.session_state.pop("auto_run", False)
    if (run_requested or auto_run) and clean_df is not None:
        with st.spinner("Running forecast..."):
            try:
                st.session_state["result"] = run_analysis(clean_df, settings)
                st.session_state["result_source"] = "demo" if demo_mode else "upload"
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Analysis failed: {exc}")

    result: AnalysisResult | None = st.session_state.get("result")
    current_source = "demo" if demo_mode else "upload"
    if result is not None and st.session_state.get("result_source") != current_source:
        result = None

    if result is None:
        if demo_mode:
            st.info("Press **Run analysis** in the sidebar to start the demo.")
        else:
            st.info("Upload a CSV, then press **Run analysis**.")

    tabs = st.tabs(
        ["Overview", "Data", "Forecast", "Orders", "Insights", "Copilot"]
    )

    with tabs[0]:
        _overview_tab(result, report)
    with tabs[1]:
        _data_tab(clean_df, report)
    with tabs[2]:
        if result is None:
            st.info("Run an analysis first.")
        else:
            _forecast_tab(result)
    with tabs[3]:
        if result is None:
            st.info("Run an analysis first.")
        else:
            _orders_tab(result)
    with tabs[4]:
        _insights_tab()
    with tabs[5]:
        _copilot_tab()

    st.caption("Restok - forecasting and replenishment decision support.")


if __name__ == "__main__":
    main()
