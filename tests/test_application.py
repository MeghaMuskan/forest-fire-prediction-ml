"""
Test suite for the Forest Fire Risk Prediction Flask app.

Run with:  pytest tests/ -v
"""
import pytest
from conftest import VALID_FORM_DATA


# ---------------------------------------------------------------------------
# Route availability
# ---------------------------------------------------------------------------

class TestRoutes:

    def test_index_get_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_predictdata_get_returns_200(self, client):
        """GET on /predictdata should render the empty form, not error."""
        response = client.get("/predictdata")
        assert response.status_code == 200

    def test_unknown_route_returns_404(self, client):
        response = client.get("/this-route-does-not-exist")
        assert response.status_code == 404

    def test_predictdata_rejects_get_only_methods_correctly(self, client):
        """/predictdata should accept POST (used by the form)."""
        response = client.post("/predictdata", data=VALID_FORM_DATA)
        # Should not be a method-not-allowed error
        assert response.status_code != 405


# ---------------------------------------------------------------------------
# Prediction logic — risk category boundaries
# ---------------------------------------------------------------------------

class TestRiskCategorization:
    """
    The app buckets the FWI score into Low / Moderate / High.
    These tests pin down the exact boundary behavior so a future
    refactor can't silently shift the thresholds.
    """

    @pytest.mark.parametrize(
        "fwi_score,expected_status,expected_color",
        [
            (0, "Low Fire Risk", "low"),
            (4.99, "Low Fire Risk", "low"),
            (5, "Moderate Fire Risk", "moderate"),       # boundary: low -> moderate
            (10, "Moderate Fire Risk", "moderate"),
            (14.99, "Moderate Fire Risk", "moderate"),
            (15, "High Fire Risk", "high"),               # boundary: moderate -> high
            (18.46, "High Fire Risk", "high"),
            (40, "High Fire Risk", "high"),
        ],
    )
    def test_risk_boundaries(
        self, client, mock_model, fwi_score, expected_status, expected_color
    ):
        mock_model(fwi_score)
        response = client.post("/predictdata", data=VALID_FORM_DATA)

        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert expected_status in html
        assert f"status-{expected_color}" in html


# ---------------------------------------------------------------------------
# Gauge percentage scaling
# ---------------------------------------------------------------------------

class TestGaugeScaling:
    """
    gauge_pct scales the raw FWI score against a ceiling of 30 so the
    visual bar fill matches the 0 / 15 / 30+ scale labels in the UI —
    NOT a literal 1:1 percentage of the score.
    """

    @pytest.mark.parametrize(
        "fwi_score,expected_pct",
        [
            (0, 0.0),
            (15, 50.0),
            (18.46, 61.5),
            (30, 100.0),
            (45, 100.0),   # should cap at 100, never overflow the bar
        ],
    )
    def test_gauge_pct_calculation(self, client, mock_model, fwi_score, expected_pct):
        mock_model(fwi_score)
        response = client.post("/predictdata", data=VALID_FORM_DATA)
        html = response.get_data(as_text=True)

        assert f"width: {expected_pct}%" in html

    def test_gauge_never_exceeds_100_percent(self, client, mock_model):
        """Sanity check: even an extreme score shouldn't break the layout."""
        mock_model(999)
        response = client.post("/predictdata", data=VALID_FORM_DATA)
        html = response.get_data(as_text=True)

        assert "width: 100.0%" in html
        assert "width: 999" not in html


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:

    def test_missing_fields_default_to_zero(self, client, mock_model):
        """
        application.py uses request.form.get(field, 0) — missing fields
        shouldn't crash the route, they should default to 0.
        """
        mock_model(2.5)
        incomplete_data = {"Temperature": "30"}  # everything else missing
        response = client.post("/predictdata", data=incomplete_data)
        assert response.status_code == 200

    def test_valid_full_submission_returns_prediction(self, client, mock_model):
        mock_model(12.3)
        response = client.post("/predictdata", data=VALID_FORM_DATA)
        html = response.get_data(as_text=True)

        assert response.status_code == 200
        assert "12.3" in html
        assert "Moderate Fire Risk" in html

    def test_non_numeric_input_returns_400_or_500(self, client, mock_model):
        """
        Non-numeric form input should fail predictably (float() conversion
        error), not silently succeed with garbage data.
        """
        mock_model(10)
        bad_data = dict(VALID_FORM_DATA)
        bad_data["Temperature"] = "not-a-number"
        response = client.post("/predictdata", data=bad_data)
        assert response.status_code in (400, 500)