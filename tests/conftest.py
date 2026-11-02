"""Pytest global configuration, path setup, and shared fixtures."""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path so 'src' can be imported anywhere
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.api.main import app

@pytest.fixture(scope="session")
def client():
    """Provides a reusable FastAPI TestClient across the test session."""
    return TestClient(app)
