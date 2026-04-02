import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.Recommenders.train_gift_two_tower import (
    DEFAULT_MODEL_DIR,
    load_model_bundle,
    recommend_for_user,
    train_and_evaluate,
)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)

MODEL_BUNDLE = None
MODEL = None


def load_or_train_model():
    global MODEL_BUNDLE, MODEL

    if MODEL_BUNDLE is not None and MODEL is not None:
        return MODEL_BUNDLE, MODEL

    try:
        MODEL_BUNDLE, MODEL = load_model_bundle(MODEL_DIR)
        print(f"Loaded model bundle from {MODEL_DIR}")
    except FileNotFoundError:
        print(f"Model bundle not found at {MODEL_DIR}. Training a fresh model.")
        train_and_evaluate(save_model=True, model_dir=MODEL_DIR)
        MODEL_BUNDLE, MODEL = load_model_bundle(MODEL_DIR)
        print(f"Finished training and loaded model bundle from {MODEL_DIR}")

    return MODEL_BUNDLE, MODEL


class RecommendationHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}

        raw_body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw_body)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/":
            self._send_json(
                {
                    "message": "Gift two-tower recommender",
                    "predict_path": "/predict",
                    "health_path": "/health",
                }
            )
            return

        if self.path == "/health":
            self._send_json({"status": "ok", "model_dir": MODEL_DIR})
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/predict":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            json_payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON"}, status=400)
            return

        user_id = json_payload.get("user_id") or json_payload.get("user")
        top_k = int(json_payload.get("top_k", 5))

        if not user_id:
            self._send_json({"error": "Missing required field: user_id"}, status=400)
            return

        if top_k < 1:
            self._send_json({"error": "top_k must be at least 1"}, status=400)
            return

        bundle, model = load_or_train_model()
        recommendations = recommend_for_user(model, bundle, user_id, top_k=top_k)
        self._send_json(
            {
                "user_id": user_id,
                "top_k": top_k,
                "model_dir": MODEL_DIR,
                "recommendations": recommendations,
            }
        )


if __name__ == "__main__":
    load_or_train_model()
    server = ThreadingHTTPServer((HOST, PORT), RecommendationHandler)
    print(f"Serving recommendations on http://{HOST}:{PORT}")
    server.serve_forever()
