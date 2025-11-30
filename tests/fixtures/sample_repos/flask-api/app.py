"""Flask API application."""

import os
from flask import Flask, jsonify

app = Flask(__name__)

# Configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/api/v1/status")
def status():
    """Status endpoint."""
    return jsonify({
        "version": "1.0.0",
        "database": "connected" if DATABASE_URL else "not configured",
        "cache": "connected" if REDIS_URL else "not configured",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
