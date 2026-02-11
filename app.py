from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model, scaler, model_name = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        plays = int(request.form["plays"])
        playing = int(request.form["playing"])
        wishlist = int(request.form["wishlist"])

        engagement = playing / (plays + 1)

        X_input = np.array([[playing, wishlist, plays, engagement]])

        if model_name == "Linear Regression":
            X_input = scaler.transform(X_input)

        prediction = int(model.predict(X_input)[0])

    return render_template(
        "index.html",
        prediction=prediction,
        model_name=model_name
    )


if __name__ == "__main__":
    app.run(debug=True)
