#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root, no matter where this script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Restok: Autonomous Retail Replenishment Platform ==="

# Prefer the project virtual environment, then fall back to system Python.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    PYTHON="$(command -v python)"
fi

MODE="${1:-app}"

open_browser() {
    local url="$1"
    (
        sleep 2
        if command -v open >/dev/null 2>&1; then
            open "$url" >/dev/null 2>&1
        elif command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" >/dev/null 2>&1
        fi
    ) &
}

case "$MODE" in
    app|streamlit)
        echo "Starting the Streamlit app at http://localhost:8501 ..."
        open_browser "http://localhost:8501"
        exec "$PYTHON" -m streamlit run streamlit_app.py \
            --server.port 8501 --server.headless true
        ;;

    server|api)
        echo "Starting the API and dashboard at http://localhost:8000 ..."
        exec "$PYTHON" -m uvicorn src.api.main:app --port 8000 --reload
        ;;

    test)
        shift || true
        echo "Running the test suite..."
        exec "$PYTHON" -m pytest tests/ -v "$@"
        ;;

    install)
        if [ ! -x ".venv/bin/python" ]; then
            echo "Creating virtual environment in .venv ..."
            python3 -m venv .venv
        fi
        echo "Installing dependencies..."
        .venv/bin/python -m pip install --upgrade pip
        .venv/bin/python -m pip install -r requirements.txt
        echo "Done. Start the app with: ./run.sh"
        ;;

    pipeline)
        echo "Running the end-to-end data and model pipeline (Phases 1-8)..."
        "$PYTHON" -m src.data.loader
        "$PYTHON" -m src.data.cleaner
        "$PYTHON" -m src.data.preprocessor
        "$PYTHON" -m src.simulation.supply_sim
        "$PYTHON" -m src.features.feature_pipeline
        "$PYTHON" -m src.models.train
        "$PYTHON" -m src.models.censoring_study
        "$PYTHON" -m src.policy.simulator
        echo "Pipeline complete. Starting the Streamlit app..."
        open_browser "http://localhost:8501"
        exec "$PYTHON" -m streamlit run streamlit_app.py \
            --server.port 8501 --server.headless true
        ;;

    help|-h|--help)
        cat <<'EOF'
Usage: ./run.sh [mode]

Modes:
  app        Start the Streamlit app (default)  -> http://localhost:8501
  server     Start the FastAPI API and dashboard -> http://localhost:8000
  test       Run the automated test suite
  install    Create .venv and install dependencies
  pipeline   Rebuild data, train models, then start the app
  help       Show this message

Examples:
  ./run.sh
  ./run.sh app
  ./run.sh server
  ./run.sh test
EOF
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Run './run.sh help' to see the available modes."
        exit 1
        ;;
esac
