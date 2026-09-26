"""Flask web application for the Financial Advisor Bot.

Phase 4.2 a thin HTTP layer over src/web/service.py. It does NOT contain any
ML logic; it just exposes the service over HTTP and serves the frontend.

Routes
------
  GET  /                      -> the single-page frontend.
  GET  /api/recommend/<tkr>   -> JSON recommendation for a ticker.

The /api endpoint returns the service's to_dict() output, or a JSON error with
an appropriate HTTP status if the ticker is invalid. Keeping the API and the
page separate means the frontend is just a client of the API clean, testable,
and easy to demonstrate.

Run it
------
    python -m src.web.app
then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import logging


from flask import Flask, jsonify, render_template

from src.web.service import TickerError, get_recommendation

logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Serve the single-page frontend."""
    return render_template("index.html")


@app.route("/api/recommend/<ticker>")
def api_recommend(ticker: str):
    """Return a JSON recommendation for a ticker.

    On success: 200 with the recommendation payload.
    On a bad/empty ticker or insufficient data: 400 with an error message.
    On anything unexpected: 500 with a generic message (details are logged).
    """
    try:
        rec = get_recommendation(ticker)
        return jsonify(rec.to_dict())
    except TickerError as e:
        # Expected, user-facing problem (bad symbol, too little data).
        return jsonify({"error": str(e)}), 400
    except Exception:  # noqa: BLE001 - last-resort guard so the API never 500s silently
        logger.exception("Unexpected error generating recommendation for %s", ticker)
        return (
            jsonify(
                {
                    "error": "Something went wrong while analysing that ticker. "
                    "Please try again."
                }
            ),
            500,
        )


if __name__ == "__main__":
    # Development server only. debug=True gives helpful errors while building;
    # turn it off for any real deployment.
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True, host="127.0.0.1", port=5000)