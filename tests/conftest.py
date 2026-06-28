import sys
import os
import pytest

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
    Replaces the app's model_service.predict() with a deterministic stand-in
    so tests don't depend on the actual trained model's output — we're
    testing the Flask routing/classification logic, not the ML model itself.
    """
    class FakeModelService:
        def predict(self, features):
            return self.fixed_value

    fake_service = FakeModelService()

    def _set_prediction(value):
        fake_service.fixed_value = value
        monkeypatch.setattr(app_module, "model_service", fake_service)

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