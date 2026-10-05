#!/usr/bin/env bash
set -e

# Restok quick runner script
echo "=== Restok: Autonomous Retail Replenishment Platform ==="

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parse argument
MODE=${1:-server}

if [ "$MODE" = "test" ]; then
    echo "Running automated test suite..."
    pytest tests/ -v
elif [ "$MODE" = "pipeline" ]; then
    echo "Running complete end-to-end pipeline (Phases 1-8)..."
    python -m src.data.loader
    python -m src.data.cleaner
    python -m src.data.preprocessor
    python -m src.simulation.supply_sim
    python -m src.features.feature_pipeline
    python -m src.models.train
    python -m src.models.censoring_study
    python -m src.policy.simulator
    echo "Pipeline complete! Launching server..."
    uvicorn src.api.main:app --port 8000 --reload
elif [ "$MODE" = "server" ]; then
    echo "Starting Restok API server & Dashboard at http://localhost:8000 ..."
    uvicorn src.api.main:app --port 8000 --reload
else
    echo "Usage: ./run.sh [server | test | pipeline]"
fi
