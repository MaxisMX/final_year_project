# Plainstock

A free web tool that gives a young, non-technical investor a BUY or SELL read on a US stock, with a plain-English explanation and a price chart.

Final project for CM3070 (University of London), based on **Project Idea 4.2: Financial Advisor Bot** from CM3020 Artificial Intelligence.

> **This is an educational tool, not financial advice.** None of the models evaluated in this project reliably predicted short-term price direction. See the headline finding below.

## Headline finding

Four model architectures — Random Forest, LSTM, GRU and a CNN-LSTM hybrid — were compared using walk-forward validation across five folds on SPY, MSFT and TSLA. **None beat a majority-class baseline on any stock.** The highest score, the CNN-LSTM's 0.619 on SPY, sat just below the baseline of 0.620 because the model predicted BUY every day. A wider check extended the Random Forest to ten stocks across several sectors; it failed to beat the baseline on all ten.

The full evaluation is in Chapter 5 of the final report.

## Project structure

```
src/
├── data/        fetching and caching daily price data (yfinance)
├── features/    ten technical indicators and the BUY/SELL target
├── models/      Random Forest, LSTM, GRU, CNN-LSTM and the comparison framework
├── backtest/    walk-forward validation and the backtesting engine
├── explain/     SHAP analysis and the plain-English generator
├── web/         service layer and Flask application
├── wider_checks.py   ten-stock check, cost sensitivity, rule baseline (report §5.8)
└── equity_curve.py   backtest equity-curve figure (report Figure 7)
tests/           unit tests, mirroring the src/ structure
```

## Setup

Requires Python 3.10 or later.

```bash
git clone <REPO-URL>
cd final_year_project
python -m venv env
env\Scripts\activate            # Windows
# source env/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Price data is downloaded from Yahoo Finance on first use and cached in `data/raw/`.

## Run the web app

```bash
python -m src.web.app
```

Then open http://127.0.0.1:5000 and enter a ticker such as `MSFT`. The first lookup for a ticker trains a model and can take 30–60 seconds; later lookups for the same ticker reuse the cached model.

## Run the tests

```bash
pytest
```

The suite includes indicator calculations checked against hand-worked examples (for instance RSI against Wilder's published example, approximately 70.53) and tests guarding against train/test leakage in the sequence windowing.

## Reproduce the results

| Report section | Command |
|---|---|
| §5.3 walk-forward validation (Random Forest, MSFT) | `python -m src.backtest.walk_forward` |
| §5.5 backtest against buy-and-hold | `python -m src.backtest.engine` |
| §5.8 wider checks (Tables 2–5) | `python -m src.wider_checks` |
| Figure 7, equity curves | `python -m src.equity_curve` |

Results depend on the date the data is downloaded. The backtest uses a chronological split, so additional data moves the test period — Section 5.8 of the report discusses how much this changes the backtest result. Random seeds are fixed, but TensorFlow retains some non-determinism, so neural-model figures may vary slightly between runs.

## Branches

- **`master`** — the submitted, working application. The recommendation comes from the LSTM; the explanation beneath it is a rule-based reading of the indicators.
- **`explain-fix`** — a revised explanation pipeline built in response to feedback on the draft report. It serves the Random Forest, derives the explanation from SHAP values for the exact prediction shown, and withholds any explanation when the model has collapsed to a single class. It works from the command line (`python -m src.web.service MSFT`) but was not integrated into the web interface before submission. See `BRANCH_NOTES.md` on that branch and Section 4.7 of the report.