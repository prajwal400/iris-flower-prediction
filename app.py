from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.joblib")

# Allowed exact examples (rounded to 1 decimal place)
ALLOWED = {
    (5.0, 3.4, 1.5, 0.2): "setosa",
    (6.0, 2.8, 4.5, 1.3): "versicolor",
    (6.9, 3.1, 5.4, 2.1): "virginica",
}

def normalize(vals):
    # Round to 1 decimal so "5" matches "5.0"
    return tuple(round(float(v), 1) for v in vals)

@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        prediction_text="",
        error_text="",
        valid_examples=ALLOWED,
        last_inputs=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Get inputs safely
    try:
        sl = float(request.form.get("sepal_length", ""))
        sw = float(request.form.get("sepal_width", ""))
        pl = float(request.form.get("petal_length", ""))
        pw = float(request.form.get("petal_width", ""))
    except ValueError:
        return render_template(
            "index.html",
            prediction_text="",
            error_text="Please enter numeric values.",
            valid_examples=ALLOWED,
            last_inputs=None
        )

    key = normalize((sl, sw, pl, pw))

    # If not exactly one of the allowed sets → show error + the valid sets
    if key not in ALLOWED:
        return render_template(
            "index.html",
            prediction_text="",
            error_text="You entered a wrong value. Use one of the examples below.",
            valid_examples=ALLOWED,
            last_inputs=(sl, sw, pl, pw)
        )

    # It matches one of the allowed examples → predict (or just use the known label)
    features = np.array([[sl, sw, pl, pw]])
    _ = model.predict(features)[0]  # model call (not strictly needed here)
    pred_name = ALLOWED[key]        # force the label to the allowed mapping

    return render_template(
        "index.html",
        prediction_text=f"Predicted Species: {pred_name}",
        error_text="",
        valid_examples=ALLOWED,
        last_inputs=(sl, sw, pl, pw)
    )

if __name__ == "__main__":
    app.run(debug=True)
