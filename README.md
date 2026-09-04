# Regime-Aware Sector Rotation

Research framework for allocating across US sector ETFs using a causal macro-regime filter and a purged, walk-forward learning-to-rank model.

> This repository is research, not investment advice. Historical results are not evidence of future performance.

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CI](https://github.com/yuvipaloozie/Regime-Aware-Sector-Trading/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Explore the deployed interface at [regime-aware-sector-trading.streamlit.app](https://regime-aware-sector-trading.streamlit.app/).

![Regime-aware sector rotation dashboard](assets/trading-dash.png)

## Background and motivation

Financial time series are non-stationary: their means, variances, and correlations change across expansion, transition, contraction, and crisis periods. This project tests whether identifying those environments can improve sector allocation while controlling downside risk.

- **The states:** latent macro and market conditions are inferred from returns, volatility, the yield curve, financial stress, and inflation.
- **The rotation:** sectors have different sensitivities to those conditions, so the strategy ranks them cross-sectionally rather than forecasting absolute market direction.
- **The engineering premise:** causal data handling and reproducible execution matter more than adding model complexity.

## What the system does

1. Downloads adjusted daily ETF prices and point-in-time initial-release macro observations.
2. Builds stationary weekly market, volatility, yield-curve, stress, and inflation features.
3. Refits a Gaussian HMM in rolling windows and applies a forward-only state filter.
4. Trains an expanding-window `XGBRanker` with a four-week purge/embargo.
5. Selects the top-ranked sectors with a turnover buffer.
6. Allocates unused risk budget to the strongest positive-trend defensive asset among SHY, IEF, and TLT, otherwise cash.
7. Executes signals on a later trading date and marks the portfolio to market daily.

## Backtest-integrity guarantees

- **Causal HMM inference:** later observations in a prediction block cannot change an earlier state.
- **Purged labels:** a training label is excluded unless its full forward-return horizon is known before the test signal.
- **Live inference rows:** the latest rows can be scored even though their future-return labels do not yet exist.
- **Point-in-time macro data:** FRED initial releases are indexed by first-availability date rather than revised observation history.
- **Explicit demo isolation:** synthetic data is available only through `--demo`; a network failure never silently changes a real run into a synthetic one.
- **Adjusted and return-spliced prices:** pre-inception proxies are normalized at the ETF switch date instead of filling incompatible price levels.
- **Delayed execution:** signal dates and execution dates are stored separately.
- **Daily valuation:** drawdown and risk statistics include intra-rebalance-period moves.
- **Artifact provenance:** each run writes `data/run_manifest.json` with sources, model diagnostics, costs, and date ranges.

## Model architecture and visual evidence

### 1. Macro regime identification

An anchored Gaussian HMM sorts fitted states by volatility so state numbering remains interpretable. The updated implementation uses causal forward probabilities: a later observation cannot revise an earlier out-of-sample state. Exposure is probability-weighted rather than driven by a single brittle hard label.

The following figures are the original project artifacts and remain useful illustrations of the model. They are historical snapshots, not regenerated proof of the revised pipeline.

![Historical inferred regime timeline](assets/regime-history.png)

![Historical HMM state-transition matrix](assets/regime_matrix.png)

### 2. Cross-sectional sector ranking

The second stage uses `XGBRanker` to order the eleven sector ETFs by expected four-week relative alpha. Training observations are grouped by date and subjected to a strict purge/embargo. The newest unlabeled rows can still be scored for live inference.

![Historical XGBoost feature importance](assets/xgboostimp.png)

### 3. Portfolio construction and execution

The top-ranked sectors are selected with a turnover buffer. Regime probabilities scale equity exposure, while the remaining budget is allocated to the strongest positive-trend defensive asset among SHY, IEF, and TLT—or to cash when none qualifies. Signals execute on a later market date and are valued daily.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python dependencies are pinned to the versions exercised by CI.

## Running with real data

Create a FRED API key, then expose it to the process:

```bash
export FRED_API_KEY="your-key"
python regime_sector_rotation/main.py --force-refresh
```

PowerShell:

```powershell
$env:FRED_API_KEY = "your-key"
python regime_sector_rotation/main.py --force-refresh
```

The real-data pipeline fails closed if Yahoo Finance or point-in-time FRED ingestion is unavailable.

## Explicit demo mode

```bash
python regime_sector_rotation/main.py --demo --force-refresh
```

Demo artifacts are marked as synthetic in their cache metadata and run manifest. Do not compare demo performance with real benchmarks.

## What is backtest, recent signal, and live testing?

| Mode | What it does | Current status |
| :--- | :--- | :--- |
| Historical backtest | Walk-forward model fitting, delayed execution, costs, and daily mark-to-market evaluation over past data | Implemented |
| Recent research signal | Refreshes data through the configured end date (`today` by default) and scores the latest rows even though future labels do not exist | Implemented; requires `FRED_API_KEY` for a real-data refresh |
| Simulated execution replay | Replays the final allocation changes through `SimulatedTradeManager` and writes `trade_log.csv` | Implemented; this is not a broker-connected paper account |
| Scheduled paper trading | Runs automatically on a schedule, submits orders to a broker sandbox, reconciles fills, and monitors failures | Not yet implemented |
| Live trading | Places real-money orders | Not implemented |

The Streamlit app reads the generated artifacts; it does not run continuously and does not prove that an order was submitted or filled. Its status banner identifies legacy, demo, or real-data artifacts and explicitly labels executions as simulated.

## Outputs

- `data/backtest_equity.csv`: daily strategy, equal-weight, SPY, and momentum wealth paths.
- `data/backtest_weights.csv`: signal-date target weights.
- `data/trade_log.csv`: delayed simulated executions and allocation records.
- `data/run_manifest.json`: data provenance, ranking diagnostics, cost sensitivity, and performance metrics.
- `ML_Sector_Rotation_Tearsheet.html`: QuantStats report based on daily returns.

## Evaluation

Each run reports:

- CAGR, annualized Sharpe, daily maximum drawdown, and annualized turnover;
- mean out-of-sample Spearman rank information coefficient;
- top-N realized-sector hit rate;
- performance at 0, 5, 10, and 20 basis points per unit of turnover;
- comparisons with SPY, equal-weight sectors, and a simple momentum baseline.

Previously committed headline statistics were produced by an earlier pipeline and are not treated as current evidence. Regenerate artifacts after configuring point-in-time data, then assess the new manifest and tearsheet.

### Original project results — legacy snapshot

These are the figures originally presented with the project. They preserve the original research narrative and screenshots, but they came from the pre-integrity-refactor pipeline and have **not** been reproduced by the revised causal pipeline. They must not be presented as current validated performance.

| Metric | S&P 500 (SPY) | Static equal weight | Original regime-aware run |
| :--- | ---: | ---: | ---: |
| Sharpe ratio | 0.65 | 0.61 | **0.88** |
| Maximum drawdown | -54.3% | -51.2% | **-32.4%** |
| Annualized volatility | 18.2% | 17.5% | **12.1%** |

The revised run manifest is now the canonical source for performance claims because it records data provenance, evaluation dates, cost assumptions, ranking diagnostics, and sensitivity results.

## Interpretability

- **State clarity:** state 0 represents the lowest-volatility fitted state and state 3 the highest-volatility fitted state, with posterior probabilities retained for every observation.
- **Feature attribution:** walk-forward feature importances show how momentum, volatility, beta, rate sensitivity, and regime probabilities influence sector rankings.
- **Decision traceability:** signal weights, delayed execution weights, transactions, regime probabilities, costs, and equity paths are exported separately.

## Repository layout

```text
regime_sector_rotation/
├── config/settings.yaml
├── src/
│   ├── pipeline_ingest.py
│   ├── features.py
│   ├── model_hmm.py
│   ├── model_strategy.py
│   └── backtester.py
├── tests/test_components.py
├── main.py
└── app.py
```

## Remaining research limitations

- Yahoo Finance is convenient research data, not an institutional market-data feed.
- Initial-release FRED data reduces revision leakage but release-time granularity may still require a vendor-grade economic calendar.
- HMM state meaning can drift even when states are ordered by volatility; monitor emission centers and transition stability.
- Model and portfolio hyperparameters still require nested walk-forward selection before any claim of optimality.
- Capacity, taxes, borrow constraints, opening-auction fills, and market impact are not modeled.
- T-bills and Treasury ETFs can behave differently from cash during stress; the defensive sleeve must be monitored rather than assumed safe.

## Production and portfolio roadmap

### Recruiter-ready engineering

- Package the engine behind typed interfaces and a CLI, with Ruff, mypy, pytest coverage thresholds, pre-commit hooks, and generated API documentation.
- Add deterministic unit fixtures, golden backtest tests, property tests for weight/risk invariants, and integration tests for provider schema changes.
- Containerize the app and pipeline, add environment-specific configuration, and publish reproducible build artifacts from CI.
- Track experiments, parameters, datasets, and model versions with an experiment registry rather than relying on notebooks or overwritten CSV files.
- Add an architecture diagram, model card, data dictionary, decision log, and a concise case study explaining what changed after leakage was removed.

### Production trading controls

- Replace research feeds with licensed point-in-time market and macro data carrying release and correction timestamps.
- Persist signals, orders, fills, positions, cash, and model versions in a transactional database with idempotency keys and an immutable audit trail.
- Add a scheduler, broker paper-account adapter, order reconciliation, retry policy, alerts, health checks, and a manual kill switch.
- Monitor missing/stale data, feature drift, regime drift, prediction dispersion, turnover, exposure, slippage, drawdown, and divergence between expected and actual fills.
- Separate research, paper, and production credentials and require an approval gate before promoting a model version.


## Tests

```bash
python -m compileall regime_sector_rotation
python -m unittest discover -s regime_sector_rotation/tests -v
```

CI runs these checks on Python 3.10 and 3.12.

## License

MIT
