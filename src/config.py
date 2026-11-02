"""Configuration loader for Restok."""
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Loads yaml configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / cfg["data"]["raw_dir"]
PROCESSED_DIR = PROJECT_ROOT / cfg["data"]["processed_dir"]
STORES: List[str] = cfg["data"]["stores"]
CATEGORY: str = cfg["data"]["category"]
START_D: int = cfg["data"]["start_d"]
END_D: int = cfg["data"]["end_d"]
FORECAST_HORIZON: int = cfg["data"]["forecast_horizon"]
SKUS_PER_DEPT: Dict[str, int] = cfg["data"]["skus_per_dept"]
SUPPLY_CHAIN_CFG: Dict[str, Any] = cfg["supply_chain"]
