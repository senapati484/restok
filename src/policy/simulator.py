"""Phase 8: Replenishment Policy Simulation for Restok.

Compares three replenishment policies across 20 stochastic seeds over the 14-day test horizon:
1. Manager Heuristic (static multiplier of 7-day rolling mean)
2. Point Forecast Policy (LightGBM Tweedie with promo buffer and perishable cap)
3. Quantile Ordering Policy (Newsvendor critical ratio tau with case pack rounding)

Generates:
- Comprehensive KPI summary with 95% confidence intervals
- Service-level vs. cost tradeoff curve
- Honest evaluation of whether the 18% stockout and 10% inventory reduction targets were met
Outputs to data/processed/policy_kpi_summary.csv and data/processed/policy_tradeoff_curve.csv.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def simulate_single_seed(
    df: pd.DataFrame,
    policy_name: str,
    seed: int = 42,
    service_level_bias: float = 0.0,
) -> Dict[str, float]:
    """Runs day-by-day inventory simulation over the 14-day horizon for one random seed."""
    rng = np.random.default_rng(seed)

    total_sales = 0
    total_demand = 0
    total_lost = 0
    stockout_events = 0
    total_events = 0
    daily_inv_units = []
    holding_costs = []
    stockout_penalties = []

    # Daily holding cost rate based on 20% annual
    daily_holding_rate = 0.20 / 365.0

    for (store, item), group in df.groupby(["store_id", "item_id"], sort=False):
        lead_time = int(group["lead_time"].iloc[0])
        case_pack = int(group["case_pack"].iloc[0])
        shelf_life = int(group["shelf_life"].iloc[0])
        unit_cost = float(group["unit_cost"].iloc[0])
        unit_margin = float(group["unit_margin"].iloc[0])
        penalty_per_unit = unit_margin + unit_cost * 0.5

        # Initialize inventory with seed variability
        base_init = float(group["rolling_mean_7_lag14"].iloc[0]) * (lead_time + 2)
        on_hand = max(case_pack, int(rng.normal(base_init, max(1.0, base_init * 0.15))))

        pipeline = {}  # delivery_day -> qty

        dates = group["date"].values
        demands = group["true_demand"].values
        pt_preds = group["pred_point"].values
        rm7_preds = group["rolling_mean_7_lag14"].values
        rm_std = group["rolling_std_7_lag14"].values
        is_promo = group["is_promotion"].values

        # Quantile prediction selection based on critical ratio
        cr = float(group["critical_ratio"].iloc[0])
        if cr <= 0.78:
            q_preds = group["pred_q_75"].values
        elif cr <= 0.88:
            q_preds = group["pred_q_85"].values
        else:
            q_preds = group["pred_q_90"].values

        for t in range(len(dates)):
            # 1. Incoming order arrivals
            arrivals = pipeline.pop(t, 0)
            available = on_hand + arrivals

            # 2. Demand fulfillment
            demand = int(demands[t])
            sold = min(available, demand)
            lost = demand - sold

            total_sales += sold
            total_demand += demand
            total_lost += lost
            total_events += 1

            if lost > 0 or (available == 0 and demand > 0):
                stockout_events += 1

            on_hand = available - sold
            daily_inv_units.append(on_hand)

            holding_cost = on_hand * unit_cost * daily_holding_rate
            penalty = lost * penalty_per_unit
            holding_costs.append(holding_cost)
            stockout_penalties.append(penalty)

            # 3. Replenishment ordering
            inv_pos = on_hand + sum(pipeline.values())

            if policy_name == "manager_heuristic":
                # Static multiplier on 7-day trailing mean
                rop = 1.30 * rm7_preds[t] * lead_time
                target_stock = 1.35 * rm7_preds[t] * (lead_time + 2)
                if inv_pos <= rop:
                    order_needed = max(0, target_stock - inv_pos)
                    order_qty = int(np.ceil(order_needed / case_pack) * case_pack)
                    if order_qty > 0:
                        pipeline[t + lead_time] = pipeline.get(t + lead_time, 0) + order_qty

            elif policy_name == "point_forecast":
                # Point forecast + historical std safety buffer + promo boost
                safety_stock = 1.15 * rm_std[t] * np.sqrt(lead_time)
                promo_mult = 1.25 if is_promo[t] == 1 else 1.0
                rop = pt_preds[t] * lead_time * promo_mult + safety_stock
                target_stock = pt_preds[t] * (lead_time + 2) * promo_mult + safety_stock
                # Perishable shelf life cap
                target_stock = min(target_stock, pt_preds[t] * shelf_life)
                if inv_pos <= rop:
                    order_needed = max(0, target_stock - inv_pos)
                    order_qty = int(np.ceil(order_needed / case_pack) * case_pack)
                    if order_qty > 0:
                        pipeline[t + lead_time] = pipeline.get(t + lead_time, 0) + order_qty

            elif policy_name == "quantile_ordering":
                # Newsvendor critical ratio quantile with optional tuning bias
                q_adj = q_preds[t] * (1.0 + service_level_bias)
                rop = q_adj * lead_time
                target_stock = q_adj * (lead_time + 2)
                # Perishable shelf life cap
                target_stock = min(target_stock, pt_preds[t] * shelf_life)
                if inv_pos <= rop:
                    order_needed = max(0, target_stock - inv_pos)
                    order_qty = int(np.ceil(order_needed / case_pack) * case_pack)
                    if order_qty > 0:
                        pipeline[t + lead_time] = pipeline.get(t + lead_time, 0) + order_qty

    service_level = (total_sales / total_demand * 100.0) if total_demand > 0 else 0.0
    stockout_rate = (stockout_events / total_events * 100.0) if total_events > 0 else 0.0
    avg_inv = float(np.mean(daily_inv_units))
    tot_holding = float(np.sum(holding_costs))
    tot_penalty = float(np.sum(stockout_penalties))
    tot_cost = tot_holding + tot_penalty

    return {
        "service_level_pct": service_level,
        "stockout_rate_pct": stockout_rate,
        "avg_inventory_units": avg_inv,
        "holding_cost_usd": tot_holding,
        "stockout_penalty_usd": tot_penalty,
        "total_cost_usd": tot_cost,
    }


def run_policy_benchmark(num_seeds: int = 20) -> pd.DataFrame:
    """Simulates the 3 policies across 20 stochastic seeds and generates KPI statistics."""
    forecasts_path = PROCESSED_DIR / "test_forecasts.parquet"
    assert forecasts_path.exists(), f"Missing {forecasts_path}. Run Phase 6 first."

    logger.info("Loading test forecasts: %s", forecasts_path)
    df = pd.read_parquet(forecasts_path)
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    policies = ["manager_heuristic", "point_forecast", "quantile_ordering"]
    seed_records = []

    logger.info("Simulating %d policies over %d stochastic seeds...", len(policies), num_seeds)
    for pol in policies:
        for seed in range(1, num_seeds + 1):
            res = simulate_single_seed(df, policy_name=pol, seed=seed)
            seed_records.append({"Policy": pol, "Seed": seed, **res})

    runs_df = pd.DataFrame(seed_records)

    # Compute summary statistics (Mean, Std, 95% Confidence Interval)
    summary_rows = []
    for pol in policies:
        sub = runs_df[runs_df["Policy"] == pol]
        row = {"Policy": pol}
        for metric in [
            "service_level_pct",
            "stockout_rate_pct",
            "avg_inventory_units",
            "holding_cost_usd",
            "stockout_penalty_usd",
            "total_cost_usd",
        ]:
            vals = sub[metric].values
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1)
            ci_half = 1.96 * (std_val / np.sqrt(len(vals)))
            row[f"{metric}_mean"] = round(mean_val, 2)
            row[f"{metric}_ci_low"] = round(mean_val - ci_half, 2)
            row[f"{metric}_ci_high"] = round(mean_val + ci_half, 2)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_path = PROCESSED_DIR / "policy_kpi_summary.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("Policy KPI Summary saved to %s", out_path)

    # 4. Target Verification:
    heur_stockout = summary_df.loc[summary_df["Policy"] == "manager_heuristic", "stockout_rate_pct_mean"].iloc[0]
    quant_stockout = summary_df.loc[summary_df["Policy"] == "quantile_ordering", "stockout_rate_pct_mean"].iloc[0]
    stockout_reduction_pct = ((heur_stockout - quant_stockout) / heur_stockout) * 100.0

    heur_inv = summary_df.loc[summary_df["Policy"] == "manager_heuristic", "avg_inventory_units_mean"].iloc[0]
    quant_inv = summary_df.loc[summary_df["Policy"] == "quantile_ordering", "avg_inventory_units_mean"].iloc[0]
    inv_reduction_pct = ((heur_inv - quant_inv) / heur_inv) * 100.0

    logger.info("=== TARGET VERIFICATION REPORT ===")
    logger.info(
        "Stockout Rate: Heuristic=%.2f%% -> Quantile=%.2f%% (Reduction: %.2f%% | Target: 18%%)",
        heur_stockout,
        quant_stockout,
        stockout_reduction_pct,
    )
    logger.info(
        "Average Inventory: Heuristic=%.1f units -> Quantile=%.1f units (Reduction: %.2f%% | Target: 10%%)",
        heur_inv,
        quant_inv,
        inv_reduction_pct,
    )

    # 5. Service-Level vs Cost Tradeoff Curve
    logger.info("Generating Service-Level vs Cost Tradeoff Curve...")
    bias_sweep = np.linspace(-0.25, 0.35, 10)
    curve_records = []
    for b in bias_sweep:
        res = simulate_single_seed(df, policy_name="quantile_ordering", seed=42, service_level_bias=b)
        curve_records.append({
            "service_level_bias": round(b, 2),
            "service_level_pct": round(res["service_level_pct"], 2),
            "stockout_rate_pct": round(res["stockout_rate_pct"], 2),
            "avg_inventory_units": round(res["avg_inventory_units"], 1),
            "holding_cost_usd": round(res["holding_cost_usd"], 2),
            "stockout_penalty_usd": round(res["stockout_penalty_usd"], 2),
            "total_cost_usd": round(res["total_cost_usd"], 2),
        })

    curve_df = pd.DataFrame(curve_records)
    curve_out = PROCESSED_DIR / "policy_tradeoff_curve.csv"
    curve_df.to_csv(curve_out, index=False)
    logger.info("Tradeoff Curve saved to %s", curve_out)

    return summary_df


if __name__ == "__main__":
    run_policy_benchmark()
