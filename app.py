import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from predict import predict_batch, predict_fare, price_trend

BASE_DIR = Path(__file__).resolve().parent

# Flask looks for the HTML file relative to this folder.
# It works with index.html inside a "templates" folder OR right next to app.py.
app = Flask(__name__, template_folder=str(BASE_DIR))
TEMPLATE = "templates/index.html" if (BASE_DIR / "templates" / "index.html").exists() else "index.html"

AIRLINES = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
TIMES = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
STOPS = ["zero", "one", "two_or_more"]
CLASSES = ["Economy", "Business"]


def read_flight(data):
    """Validate the JSON sent by the page and return clean values."""
    flight = {
        "airline": data.get("airline"),
        "source": data.get("source"),
        "dest": data.get("dest"),
        "dep": data.get("dep"),
        "arr": data.get("arr"),
        "stops": data.get("stops"),
        "cls": data.get("cls"),
        "duration": float(data.get("duration", 0)),
        "days_left": int(data.get("days_left", 0)),
    }
    if flight["source"] == flight["dest"]:
        raise ValueError("Choose different cities for From and To.")
    if flight["airline"] not in AIRLINES or flight["source"] not in CITIES \
            or flight["dest"] not in CITIES or flight["dep"] not in TIMES \
            or flight["arr"] not in TIMES or flight["stops"] not in STOPS \
            or flight["cls"] not in CLASSES:
        raise ValueError("One of the selected options is not valid.")
    if not 0.5 <= flight["duration"] <= 50:
        raise ValueError("Duration must be between 0.5 and 50 hours.")
    if not 1 <= flight["days_left"] <= 49:
        raise ValueError("Days left must be between 1 and 49.")
    return flight


@app.get("/")
def home():
    return render_template(
        TEMPLATE,
        airlines=AIRLINES, cities=CITIES, times=TIMES, stops=STOPS, classes=CLASSES,
    )


@app.post("/api/fare")
def api_fare():
    try:
        f = read_flight(request.get_json(force=True))
        price = predict_fare(f["airline"], f["source"], f["dep"], f["stops"],
                             f["arr"], f["dest"], f["cls"], f["duration"], f["days_left"])

        trend = price_trend(f["airline"], f["source"], f["dep"], f["stops"],
                            f["arr"], f["dest"], f["cls"], f["duration"])

        rows = [[a, f["source"], f["dep"], f["stops"], f["arr"], f["dest"],
                 f["cls"], f["duration"], f["days_left"]] for a in AIRLINES]
        compare = predict_batch(pd.DataFrame(rows, columns=[
            "airline", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "class", "duration", "days_left",
        ]))[["airline", "predicted_price"]].sort_values("predicted_price")

        return jsonify({
            "price": round(float(price)),
            "trend": {
                "days": trend["days_left"].tolist(),
                "prices": [round(float(p)) for p in trend["predicted_price"]],
            },
            "compare": [
                {"airline": r.airline, "price": round(float(r.predicted_price))}
                for r in compare.itertuples()
            ],
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        return jsonify({"error": "Something went wrong while predicting. Try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
