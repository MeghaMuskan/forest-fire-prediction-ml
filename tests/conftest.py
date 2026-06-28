import sys
import os
import pytest
import numpy as np

# Make sure the project root (where application.py lives) is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import application as app_module


@pytest.fixture
def app():
    """Flask app instance configured for testing."""
    app_module.app.config.update(TESTING=True, PROPAGATE_EXCEPTIONS=False)
    return app_module.app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def mock_model(monkeypatch):
    """
    Replaces the loaded ridge_model and standard_scaler with deterministic
    stand-ins so tests don't depend on the actual trained model's output —
    we're testing the Flask routing/logic, not the ML model itself.
    """
    class FakeScaler:
        def transform(self, data):
            # Pass the input straight through unscaled
            return np.array(data)

    class FakeModel:
        def __init__(self, fixed_value):
            self.fixed_value = fixed_value

        def predict(self, data):
            return np.array([self.fixed_value])

    def _set_prediction(value):
        monkeypatch.setattr(app_module, "standard_scaler", FakeScaler())
        monkeypatch.setattr(app_module, "ridge_model", FakeModel(value))

    return _set_prediction


VALID_FORM_DATA = {
    "Temperature": "32",
    "RH": "45",
    "Ws": "15",
    "Rain": "0",
    "FFMC": "85",
    "DMC": "20",
    "ISI": "8",
    "Classes": "1",
    "Region": "1",
}