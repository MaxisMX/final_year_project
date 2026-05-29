# Financial Advisor Bot

A machine learning system that analyses stock market data and produces explainable
BUY/SELL recommendations for non-technical investors, presented via a web interface.

Final Year Project for CM3020 Artificial Intelligence (University of London).

## Project Goals

- Train ML models (Random Forest baseline, LSTM main focus) on technical indicators
  to predict price movements
- Evaluate strategies rigorously using walk-forward validation against a buy-and-hold baseline
- Produce plain-English explanations of every recommendation so non-technical users
  can evaluate the advice
- Deliver a clean web interface for ticker lookup and recommendation display

## Project Structure

```
financial-advisor-bot/
├── src/
│   ├── data/        # Data fetching (yfinance) and caching
│   ├── features/    # Technical indicators (MA, RSI, MACD, etc.)
│   ├── models/      # RF baseline, LSTM, training/eval code
│   ├── backtest/    # Walk-forward validation, backtesting framework
│   ├── explain/     # SHAP + rule-based plain-English generator
│   └── web/         # Flask API and frontend
├── tests/           # pytest test suite
├── notebooks/       # Exploratory analysis (NOT for final code)
├── data/
│   ├── raw/         # Raw OHLCV data cache (gitignored)
│   └── processed/   # Engineered features (gitignored)
└── docs/
    └── devlog.md    # Development log — decisions, dead ends, dates
```

## Setup

Requires Python 3.13.

```bash
python3.13 -m venv venv
source venv/bin/activate          # macOS/Linux
# .\venv\Scripts\activate         # Windows
pip install -r requirements.txt
pytest                            # confirm scaffold works
```

### Note on Python 3.13 and TensorFlow

TensorFlow's 3.13 support is recent. If `pip install tensorflow` fails when you
reach Phase 2 (LSTM work), pin to the latest 3.13-compatible release or create a
separate Python 3.12 venv for LSTM training. The Random Forest baseline does not
need TensorFlow, so Phase 1 is unaffected.

## Development Phases

- **Phase 0 — Foundations** (this scaffold)
- **Phase 1 — Thin end-to-end slice**: one ticker, 2–3 indicators, RF baseline,
  simple backtest, CLI recommendation
- **Phase 2 — Rigour**: full indicator set, walk-forward validation, LSTM on
  identical pipeline, fair comparison
- **Phase 3 — Explainability**: SHAP on RF, rule-based plain-English generator,
  scoped LSTM interpretability
- **Phase 4 — Web interface**: Flask API, minimal frontend, Plotly charts
- **Phase 5 — Evaluation and write-up**

## A Note on Honest Evaluation

Beating the market with technical-indicator models is genuinely hard. This project
is marked on methodological rigour, not profitability. An honest result that
barely beats buy-and-hold — properly validated — is worth more than an impressive
result built on data leakage. Every claim in the final report must be supported by
the actual numbers.

## License

MIT — see [LICENSE](LICENSE).
