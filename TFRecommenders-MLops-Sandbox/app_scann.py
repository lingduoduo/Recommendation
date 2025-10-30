import logging

from flask import Flask, request, jsonify
from flask.logging import create_logger

from util.load_scann_model import predict_train_scann_model

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)


@app.route("/")
def home():
    html = "<h3>Recommend broadcasters</h3>"
    return html.format(format)


@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the recommendations for the user using the model in the specified path"""

    json_payload = request.json
    LOG.info(json_payload)
    pred = predict_train_scann_model(json_payload["user"], json_payload["path"])
    print(
        f"--- recommendations for {json_payload['user']} --- using the {json_payload['path']} model "
    )
    readable = ",".join(f"{k}:{v}" for k, v in pred.items())
    return jsonify({"recommendations": readable})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
