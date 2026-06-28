"""
Forest Fire Risk Prediction — Flask application.

Refactored into three single-responsibility classes:
- InputValidator : extracts and validates incoming form data
- ModelService   : owns the trained model + scaler, handles inference
- RiskClassifier : converts a raw FWI score into a risk category + gauge %
"""

import pickle

from flask import Flask, request, render_template


# ---------------------------------------------------------------------------
# Domain classes
# ---------------------------------------------------------------------------

class InputValidator:
    """Extracts and validates the meteorological input fields from a form."""

    FIELDS = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]

    @classmethod
    def extract(cls, form):
        """
        Pulls each expected field out of the submitted form.
        Missing fields default to 0 (matches original app behavior).
        Raises ValueError if a field can't be converted to float.
        """
        return [float(form.get(field, 0)) for field in cls.FIELDS]


class ModelService:
    """Owns the trained Ridge regression model and its fitted scaler."""

    def __init__(self, model_path, scaler_path):
        self.model = pickle.load(open(model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))

    def predict(self, features):
        """
        Scales raw input features and returns a single rounded FWI prediction.
        `features` is expected to be a flat list matching InputValidator.FIELDS order.
        """
        scaled = self.scaler.transform([features])
        raw_prediction = self.model.predict(scaled)[0]
        return round(raw_prediction, 2)


class RiskClassifier:
    """Converts a raw FWI score into a risk category, color, and gauge fill %."""

    LOW_THRESHOLD = 5
    MODERATE_THRESHOLD = 15
    GAUGE_MAX = 30  # ceiling used to scale the visual gauge bar

    def classify(self, score):
        if score < self.LOW_THRESHOLD:
            status, color = "Low Fire Risk", "low"
        elif score < self.MODERATE_THRESHOLD:
            status, color = "Moderate Fire Risk", "moderate"
        else:
            status, color = "High Fire Risk", "high"

        gauge_pct = round(min((score / self.GAUGE_MAX) * 100, 100.0), 1)

        return {"status": status, "color": color, "gauge_pct": gauge_pct}


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

application = Flask(__name__)
app = application

model_service = ModelService("models/ridge.pkl", "models/scaler.pkl")
risk_classifier = RiskClassifier()


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method != "POST":
        return render_template("home.html")

    features = InputValidator.extract(request.form)
    prediction = model_service.predict(features)
    risk = risk_classifier.classify(prediction)

    return render_template(
        "home.html",
        result=prediction,
        status=risk["status"],
        color=risk["color"],
        gauge_pct=risk["gauge_pct"],
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0")