"""Generates publication-grade figures and architecture diagrams for Restok documentation."""
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Cronus & Restok Executive Color Palette (print-friendly light canvas)
NAVY = "#1E4D8C"       # Cronus Primary Blue
GREEN = "#65A338"      # Cronus Accent Green
EMERALD = "#0F766E"    # Metric highlight teal/emerald
SLATE = "#1E293B"
MUTED = "#64748B"
BORDER = "#CBD5E1"
BG_TINT = "#F8FAFC"
AMBER = "#D97706"
ROSE = "#DC2626"
BLUE = "#2563EB"


def draw_architecture_diagram() -> Path:
    """Draws clean Phase 1-9 system architecture diagram."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, title, sub, fill="#F1F5F9", edge=NAVY, title_color=NAVY):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor=fill, edgecolor=edge, linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=title_color, family="sans-serif")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=7.0, color=SLATE, family="sans-serif")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.65",
                            color=MUTED, lw=1.3),
        )

    # Section labels
    ax.text(50, 53.5, "Restok End-to-End Pipeline Architecture (Phases 1 - 9)",
            ha="center", va="center", fontsize=11, fontweight="bold", color=SLATE)

    # Row 1: Data Engineering (Phases 1-3)
    box(3, 39, 26, 9, "Phase 1: Data Loader", "M5 FOODS (5 Stores x 60 SKUs)\nDowncast RAM -80.2%", "#EFF6FF", NAVY)
    box(37, 39, 26, 9, "Phase 2: Data Cleaner", "Wide-to-Long & Price Join\n54,000 Continuous Rows", "#EFF6FF", NAVY)
    box(71, 39, 26, 9, "Phase 3: Preprocessor", "Promotions, Holidays, Margins\nShelf Life, Lead Time, Case Pack", "#EFF6FF", NAVY)
    arrow(29.8, 43.5, 36.2, 43.5)
    arrow(63.8, 43.5, 70.2, 43.5)

    # Down connector to Row 2
    arrow(84, 38.2, 84, 32.5)

    # Row 2: Simulation & Modeling (Phases 4-7)
    box(71, 22.5, 26, 9, "Phase 4: Supply Simulator", "On-Hand & Lead-Time Receipts\n10.34% Stockouts + Conservation", "#F0FDF4", GREEN, GREEN)
    box(37, 22.5, 26, 9, "Phase 5: Feature Pipeline", "Lags t>=14, Searchsorted Holidays\n43,500 Rows x 72 Columns", "#F0FDF4", GREEN, GREEN)
    box(3, 22.5, 26, 9, "Phases 6-7: LightGBM + EM", "Optuna Tweedie + Q50..Q95\nEM Bias Recovery (-18.09%)", "#FFFBEB", AMBER, "#B45309")
    arrow(70.2, 27.0, 63.8, 27.0)
    arrow(36.2, 27.0, 29.8, 27.0)

    # Down connector to Row 3
    arrow(16, 21.7, 16, 15.8)

    # Row 3: Decision & Serving (Phases 8-9)
    box(3, 5.5, 29, 9.2, "Phase 8: Newsvendor Policy", "Critical Ratio tau = Cu/(Cu+Co)\n20-Seed Sim (-14.9% Stockouts)", "#ECFEFF", EMERALD, EMERALD)
    box(38, 5.5, 28, 9.2, "Phase 9A: FastAPI + PO Ledger", "REST APIs & JSON PO Persistence\n19/19 Pytest Validation", "#F8FAFC", NAVY, NAVY)
    box(71, 5.5, 26, 9.2, "Phase 9B: Web UI & AI Copilot", "6-Tab Executive Telemetry\nGemini 2.5 Flash Tool-Calling", "#EEF2FF", "#4338CA", "#3730A3")
    arrow(32.8, 10.1, 37.2, 10.1)
    arrow(66.8, 10.1, 70.2, 10.1)

    out = ASSETS_DIR / "fig1_architecture.png"
    plt.tight_layout(pad=0.3)
    plt.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def draw_model_benchmark(metrics_df: pd.DataFrame) -> Path:
    """Draws dual-panel Phase 6 point benchmark + quantile coverage plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel A: Point forecast WAPE (%)
    pt_df = metrics_df[metrics_df["WAPE"].notna()].copy()
    pt_df["WAPE_pct"] = pt_df["WAPE"] * 100.0
    pt_df = pt_df.sort_values("WAPE_pct", ascending=True)

    short_labels = {
        "LightGBM Tweedie (Point)": "LightGBM Tweedie",
        "Prophet Additive Decomposition": "Prophet Decomp.",
        "Baseline: 28-day Rolling Mean": "28-Day Mean",
        "Baseline: 7-day Rolling Mean": "7-Day Mean",
        "Baseline: Naive Lag-14": "Naive Lag-14",
    }
    labels = [short_labels.get(m, m) for m in pt_df["Model"]]
    colors = [EMERALD if "LightGBM" in m else "#94A3B8" for m in pt_df["Model"]]

    bars = ax1.barh(labels, pt_df["WAPE_pct"], color=colors, height=0.56, edgecolor="none")
    for bar, val in zip(bars, pt_df["WAPE_pct"]):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", ha="left", fontsize=8.5, fontweight="bold", color=SLATE)
    ax1.set_xlim(0, pt_df["WAPE_pct"].max() + 7)
    ax1.set_xlabel("Out-of-Time Test WAPE (% — Lower is Better)", fontsize=8.5, color=SLATE)
    ax1.set_title("A. Point Forecasting Error (14-Day Horizon)", fontsize=9.5, fontweight="bold", color=NAVY)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Quantile calibration coverage vs target
    q_df = metrics_df[metrics_df["Coverage"].notna()].copy()
    taus = [0.50, 0.75, 0.85, 0.90, 0.95]
    covs = (q_df["Coverage"].values * 100.0)
    targets = [t * 100.0 for t in taus]

    ax2.plot([40, 100], [40, 100], "--", color=MUTED, lw=1.2, label="Ideal Target (Coverage = tau)")
    ax2.fill_between([40, 100], [37, 97], [43, 103], color="#E2E8F0", alpha=0.45, label="+/- 3 pt Tolerance")
    ax2.plot(targets, covs, "-o", color=NAVY, lw=2.0, markersize=7, label="Empirical Coverage")

    for t_val, c_val in zip(targets, covs):
        ax2.annotate(f"{c_val:.1f}%", (t_val, c_val), textcoords="offset points",
                     xytext=(6, -12), fontsize=8, fontweight="bold", color=NAVY)

    ax2.set_xlim(45, 99)
    ax2.set_ylim(36, 100)
    ax2.set_xlabel("Nominal Quantile Target tau (%)", fontsize=8.5, color=SLATE)
    ax2.set_ylabel("Empirical Coverage P(y <= q) (%)", fontsize=8.5, color=SLATE)
    ax2.set_title("B. Conformal Quantile Calibration", fontsize=9.5, fontweight="bold", color=NAVY)
    ax2.legend(loc="lower right", fontsize=7.5, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out = ASSETS_DIR / "fig2_model_benchmark.png"
    plt.tight_layout(pad=1.0)
    plt.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def draw_sample_forecast() -> Path:
    """Draws real 28-day history + 14-day multi-quantile cone for CA_1 / FOODS_1_012."""
    feat_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_forecasts.parquet")

    store_id, item_id = "CA_1", "FOODS_1_012"
    min_test = test_df["date"].min()
    hist = feat_df[(feat_df["store_id"] == store_id) & (feat_df["item_id"] == item_id) & (feat_df["date"] < min_test)].sort_values("date").tail(28)
    fore = test_df[(test_df["store_id"] == store_id) & (test_df["item_id"] == item_id)].sort_values("date")

    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=300)
    fig.patch.set_facecolor("white")

    ax.plot(hist["date"], hist["units_sold"], "-o", color=SLATE, lw=1.6, markersize=3.5, label="Observed Units Sold (28d History)")
    ax.plot(fore["date"], fore["true_demand"], "o", color="#475569", markersize=4.5, alpha=0.75, label="Actual Ground-Truth Demand (Test)")
    ax.fill_between(fore["date"], fore["pred_q_50"], fore["pred_q_95"], color="#BFDBFE", alpha=0.55, label="Uncertainty Envelope (q50 - q95)")
    ax.plot(fore["date"], fore["pred_q_85"], "--", color=AMBER, lw=1.4, label="Upper Safety Quantile (q85)")
    ax.plot(fore["date"], fore["pred_point"], "-s", color=NAVY, lw=2.0, markersize=4.5, label="LightGBM Tweedie Point Forecast")
    ax.axvline(min_test, color=ROSE, linestyle=":", lw=1.3, label="14-Day Horizon Split (2016-05-09)")

    ax.set_title(f"14-Day Multi-Quantile Demand Forecast Cone — Store {store_id} | {item_id}", fontsize=10, fontweight="bold", color=NAVY)
    ax.set_ylabel("Daily Units", fontsize=8.5, color=SLATE)
    ax.set_xlabel("Date", fontsize=8.5, color=SLATE)
    ax.tick_params(axis="x", labelsize=7.5, rotation=20)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = ASSETS_DIR / "fig3_sample_forecast.png"
    plt.tight_layout(pad=0.8)
    plt.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def draw_censoring_and_policy(cens_df: pd.DataFrame, pol_df: pd.DataFrame, trade_df: pd.DataFrame) -> Path:
    """Draws dual-panel Phase 7 Censoring Bias + Phase 8 Policy Tradeoff Pareto curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel A: Forecast Bias (%) across censoring methods (closer to 0% is better)
    short_cens = ["1. Naive\n(Censored)", "2. Drop\nStockouts", "3. Rolling Mean\nImpute", "4. Iterative EM\nImputation"]
    bias_pct = cens_df["Forecast_Bias"].values * 100.0
    bar_cols = [ROSE, "#B91C1C", AMBER, EMERALD]

    bars = ax1.bar(short_cens, bias_pct, color=bar_cols, width=0.68)
    ax1.axhline(0, color=SLATE, lw=0.8)
    for bar, b_val, w_val in zip(bars, bias_pct, cens_df["WAPE_vs_True_Demand"].values * 100.0):
        ax1.text(bar.get_x() + bar.get_width() / 2, b_val * 0.50,
                 f"{b_val:.2f}%\nWAPE {w_val:.1f}%",
                 ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")
    ax1.set_ylim(-24.5, 1.2)
    ax1.set_ylabel("Forecast Bias vs True Demand (%)", fontsize=8.5, color=SLATE)
    ax1.set_title("A. Phase 7 Stockout Censoring Bias Recovery", fontsize=9.5, fontweight="bold", color=NAVY)
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Service Level vs Cost Frontier
    ax2.plot(trade_df["service_level_pct"], trade_df["total_cost_usd"], "-o",
             color=NAVY, lw=1.8, markersize=4.5, label="Quantile Bias Tradeoff Curve")

    pol_styles = {
        "manager_heuristic": ("Manager Heuristic", AMBER, "D", 65),
        "point_forecast": ("Point Forecast", ROSE, "s", 65),
        "quantile_ordering": ("Quantile Ordering (Restok)", EMERALD, "*", 170),
    }
    for _, row in pol_df.iterrows():
        lbl, col, marker, sz = pol_styles[row["Policy"]]
        ax2.scatter(row["service_level_pct_mean"], row["total_cost_usd_mean"],
                    c=col, marker=marker, s=sz, zorder=5, edgecolors=SLATE, linewidths=0.7, label=lbl)

    ax2.set_xlabel("Achieved Service Level / Fill Rate (%)", fontsize=8.5, color=SLATE)
    ax2.set_ylabel("14-Day Total Supply Chain Cost ($ USD)", fontsize=8.5, color=SLATE)
    ax2.set_title("B. Phase 8 Replenishment Policy Pareto Frontier", fontsize=9.5, fontweight="bold", color=NAVY)
    ax2.legend(loc="upper right", fontsize=7.3, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    out = ASSETS_DIR / "fig4_censoring_and_policy.png"
    plt.tight_layout(pad=1.0)
    plt.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def generate_all_charts() -> Dict[str, Path]:
    """Generates all technical figures directly from processed datasets."""
    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    cens_df = pd.read_csv(PROCESSED_DIR / "censoring_comparison.csv")
    pol_df = pd.read_csv(PROCESSED_DIR / "policy_kpi_summary.csv")
    trade_df = pd.read_csv(PROCESSED_DIR / "policy_tradeoff_curve.csv")

    return {
        "arch": draw_architecture_diagram(),
        "model": draw_model_benchmark(metrics_df),
        "forecast": draw_sample_forecast(),
        "policy": draw_censoring_and_policy(cens_df, pol_df, trade_df),
    }


if __name__ == "__main__":
    paths = generate_all_charts()
    for k, p in paths.items():
        print(f"Generated {k}: {p}")
