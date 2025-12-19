from flask import Flask, request, jsonify, render_template
from pricing_api import predict_price
from credit_api import predict_risk

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/credit", methods=["GET"])
def credit():
    return render_template("credit.html")

# -----------------------------
# Pricing endpoint
# -----------------------------
@app.route("/predict-price", methods=["POST"])
def predict():
    data = request.get_json()

    required = [
        "demand_index",
        "operational_cost",
        "marketing_intensity",
        "seasonality_index",
        "competition_pressure",
        "time_slot",
        "day_of_week",
    ]

    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    result = predict_price(data)
    return jsonify(result)

# -----------------------------
# Credit Risk endpoint
# -----------------------------
@app.route("/predict-risk", methods=["POST"])
def risk():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing client profile"}), 400

    try:
        result = predict_risk(data)
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
