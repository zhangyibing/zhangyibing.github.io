import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("finalized_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction == 1:
        result = "Oops! You seem to have diabetes"
    if prediction == 0:
        result = "Nice! You don't have diabetes"

    return render_template("index.html", prediction_text = "the result is {}".format(result))

if __name__ == "__main__":
    app.run(debug=True)