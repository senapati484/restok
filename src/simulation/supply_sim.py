"""Phase 4: Supply & Inventory Simulator for Restok.

Simulates on-hand inventory, lead times, order arrivals, and stockout events
to create realistic censored observable sales (units_sold) and stockout flags.
Guarantees mass conservation and ensures the stockout rate falls within 5-12%.
Outputs to data/processed/canonical_simulated.parquet.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, SUPPLY_CHAIN_CFG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def simulate_inventory_series(
    group: pd.DataFrame,
    safety_mult: float = 0.35,
    review_period_days: int = 3,
) -> pd.DataFrame:
    """Simulates daily inventory pipeline for a single Store x SKU series."""
    lead_time = int(group["lead_time"].iloc[0])
    case_pack = int(group["case_pack"].iloc[0])
    demands = group["true_demand"].values
    n = len(demands)

    # Base initial estimate on first 4 weeks of demand
    baseline_window = min(28, n)
    avg_d = max(1.0, float(np.mean(demands[:baseline_window])))
    std_d = max(1.0, float(np.std(demands[:baseline_window])))

    # Reorder point = Lead Time Demand + Safety Stock
    rop = avg_d * lead_time + safety_mult * std_d * np.sqrt(lead_time)

    # Order quantity: cover review period demand rounded up to case pack
    cycle_demand = avg_d * review_period_days
    order_qty = max(case_pack, int(np.ceil(cycle_demand / case_pack) * case_pack))

    # Initial inventory buffer (approx 7 days of demand)
    on_hand = int(avg_d * (lead_time + 4))

    pipeline = {}  # arrival_day -> quantity

    units_sold = np.zeros(n, dtype=np.int32)
    is_stockout = np.zeros(n, dtype=np.int8)
    ending_inv = np.zeros(n, dtype=np.int32)
    lost_sales = np.zeros(n, dtype=np.int32)
    receipts_arr = np.zeros(n, dtype=np.int32)

    for t in range(n):
        # 1. Receive incoming purchase orders
        receipts = pipeline.pop(t, 0)
        receipts_arr[t] = receipts
        available = on_hand + receipts

        # 2. Fulfill demand
        demand = int(demands[t])
        sold = min(available, demand)
        lost = demand - sold

        units_sold[t] = sold
        lost_sales[t] = lost
        ending = available - sold
        ending_inv[t] = ending
        on_hand = ending

        # Stockout flag
        if lost > 0 or (available == 0 and demand > 0):
            is_stockout[t] = 1

        # 3. Replenishment policy: place order if inventory position <= ROP
        inventory_position = on_hand + sum(pipeline.values())
        if inventory_position <= rop:
            delivery_day = t + lead_time
            pipeline[delivery_day] = pipeline.get(delivery_day, 0) + order_qty

    return pd.DataFrame(
        {
            "units_sold": units_sold,
            "lost_sales": lost_sales,
            "is_stockout": is_stockout,
            "on_hand_inventory": ending_inv,
            "receipts": receipts_arr,
        },
        index=group.index,
    )


def validate_simulation_results(
    df: pd.DataFrame,
    target_min_stockout: float = 0.05,
    target_max_stockout: float = 0.12,
) -> None:
    """Verifies inventory conservation and stockout rate requirements."""
    logger.info("Running simulation validation suite...")

    # 1. Stockout rate bounds
    stockout_rate = df["is_stockout"].mean()
    logger.info("Overall simulated stockout rate: %.2f%%", stockout_rate * 100)
    assert (
        target_min_stockout <= stockout_rate <= target_max_stockout
    ), f"Stockout rate {stockout_rate:.4f} is outside required bounds [{target_min_stockout}, {target_max_stockout}]"

    # 2. Conservation: units_sold <= true_demand
    violations = (df["units_sold"] > df["true_demand"]).sum()
    assert violations == 0, f"Found {violations} cases where units_sold > true_demand!"

    # 3. Lost sales reconciliation: units_sold + lost_sales == true_demand
    diff = ((df["units_sold"] + df["lost_sales"]) - df["true_demand"]).abs().sum()
    assert diff == 0, f"Units sold + lost sales does not match true demand (discrepancy: {diff})!"

    # 4. Non-negative inventory
    neg_inv = (df["on_hand_inventory"] < 0).sum()
    assert neg_inv == 0, f"Found {neg_inv} negative ending inventory instances!"

    logger.info("Simulation Validation PASSED: Conservation & stockout rate verified.")


def run_supply_simulation() -> pd.DataFrame:
    """Phase 4 main execution: runs inventory simulation across all series."""
    canonical_path = PROCESSED_DIR / "canonical.parquet"
    assert canonical_path.exists(), f"Missing {canonical_path}. Run Phase 3 first."

    logger.info("Loading canonical dataset: %s", canonical_path)
    df = pd.read_parquet(canonical_path)

    logger.info("Simulating daily supply chain dynamics across %d series...", df.groupby(["store_id", "item_id"]).ngroups)
    results = []
    for _, group in df.groupby(["store_id", "item_id"], sort=False):
        res = simulate_inventory_series(group, safety_mult=0.35, review_period_days=3)
        results.append(res)

    sim_df = pd.concat(results).sort_index()

    # Join simulated columns back to canonical dataframe
    for col in ["units_sold", "lost_sales", "is_stockout", "on_hand_inventory", "receipts"]:
        df[col] = sim_df[col]

    validate_simulation_results(df)

    out_path = PROCESSED_DIR / "canonical_simulated.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Phase 4 Complete: Simulated supply dataset saved to %s (%d rows)", out_path, len(df))

    return df


if __name__ == "__main__":
    run_supply_simulation()
