from flask import Flask, render_template, request, jsonify
import pandas as pd
from model import load_model, predict_from_model

app = Flask(__name__)

TEAMS_CSV = "teams.csv"
try:
    teams_df = pd.read_csv(TEAMS_CSV)
except Exception as e:
    teams_df = None
    print("Warning: couldn't load teams.csv:", e)

model = load_model()

@app.route("/")
def index():
    table_html = teams_df.head(200).to_html(classes="table table-striped", index=False) if teams_df is not None else "<p>No teams.csv found.</p>"
    return render_template("index.html", table_html=table_html)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or request.form
    features = data.get("features")
    if features is None:
        return jsonify({"error": "No 'features' provided."}), 400

    try:
        prediction = predict_from_model(model, features)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
