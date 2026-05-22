# Quantitative Sector Rotation via Stabilized Unsupervised Macro Regime Modeling

This repository contains an end-to-end automated data pipeline, predictive modeling framework, and execution simulator designed to implement a macro-regime-aware sector rotation investment strategy. The system models underlying market states using unsupervised learning and maps these states to cross-sectional sector alphas, mitigating common quantitative trading vulnerabilities such as state inversion and look-ahead bias.

---

## Strategy Motivation and Framework Design

Financial time-series data is inherently non-stationary, meaning its underlying statistical properties (mean, variance, and covariance) shift over time across structural market environments. Standard linear models and unconditional machine learning estimators often degrade when baseline distributions alter due to macro fluctuations. 

To address this, this framework employs a two-stage paradigm:
1. **Unsupervised Regime Identification:** Identifying latent macroeconomic states (e.g., expansionary low-volatility, contractive high-volatility) using hidden physical state spaces.
2. **Conditional Sector Allocation:** Over-weighting or under-weighting sector-specific assets based on forward-looking cross-sectional return expectations computed conditionally for the current active macro state.

---

## Pipeline Overview

The transition from exploratory research to an enterprise design is split across two core phases within the research environment:

* **Stage 1: Macro Regime Modeling (`Regime_Classifier.ipynb`):** Pulls continuous economic data (such as market volatility indices and macroeconomic proxies), transforms inputs into stationary representations, and runs an unsupervised Hidden Markov Model (HMM) to partition history into distinct operational regimes.
* **Stage 2: Conditional Sector Strategy (`Sector_Rotation_Strategy.ipynb`):** Aligns the extracted regime vector with the cross-sectional asset universe. It trains state-conditional linear and ensemble models to predict relative sector outperformance and executes a simulated rolling-window transaction ledger.

---

## System Architecture

The codebase separates concerns across a modular package structure to ensure maintainability, testing isolation, and readiness for automated execution environments (e.g., cloud cron scheduling).

```text
regime_sector_rotation/
├── config/
│   └── settings.yaml          # Hyperparameters, structural features, and asset definitions
├── data/                      # Local data caching layer (Git-ignored)
├── src/                       # Structural Core Package
│   ├── pipeline_ingest.py     # Concurrent data harvesting from FRED and Yahoo Finance
│   ├── features.py            # Stream-isolated transformation and normalization modules
│   ├── model_hmm.py           # Custom state-stabilized Hidden Markov Estimator
│   ├── model_strategy.py      # Conditional scoring and portfolio allocation logic
│   └── backtester.py          # Vector accounting and operational performance profiling
├── main.py                    # Production execution pipeline engine
└── app.py                     # Telemetry interface and trade ledger visualizer
