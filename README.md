# Regime-Aware Sector Rotation Trading Algorithm
### Dynamic Allocation via Unsupervised Macreconomic Regime Modelling

  ![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
  ![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![Domain](https://img.shields.io/badge/Domain-Quantitative_Finance-red?style=for-the-badge)
  
## Summary
This project challenges the standard static "buy-and-hold" portfolio baseline by dynamically shifting capital based on macroeconomic conditions. It benchmarks an unsupervised Hidden Markov Model (HMM) framework against static allocations while utilizing strict frameworks to avoid pitfalls of look-ahead bias and model state-inversion. To visualize and interact with the algorithm's performance in-market, you can explore more at https://regime-aware-sector-trading.streamlit.app/.

**Hypothesis:** A trading strategy grounded in unsupervised macro-regime identification can conditionally allocate sector weights to improve risk-adjusted returns (i.e. Sharpe Ratio) and reduce maximum drawdowns, provided the underlying ML pipeline mathematically enforces state stability across rolling windows.

## Background and Motivation

### Financial Context
Financial time-series data is inherently non-stationary. The underlying statistical properties (mean, variance, covariance) of the market shift dramatically depending on the macroeconomic environment. This implies that a system for one "regime" may collapse in another. 
* **The States:** The economy naturally cycles through distinct phases (e.g. boom, bust, high volatility). 
* **The Rotation:** Different industry sectors (e.g., Technology vs. Utilities) have different sensitivities to these macro phases. 

### The Engineering Problem: ML in Finance
Applying machine learning to financial data usually fails in production due to two distinct engineering flaws:
1.  **Temporal Leakage:** Standard global feature scaling injects future parameters (mean/variance) into historical training arrays. 
2.  **Label Switching (State Inversion):** Unsupervised models like HMMs classify conditions randomly per fit. Across rolling training windows, "State 0" might arbitrarily flip from representing "Low Volatility" to "High Volatility," breaking all downstream logic.

### Structural Engineering Solutions
This project hypothesizes that robust systems design is more critical than complex deep learning. We explicitly engineer solutions to map directly to operational constraints:

| Operational Risk | Domain | Pipeline Engineering |
| :--- | :--- | :--- |
| **Label Switching** | **Unsupervised Learning** | **State-Order Enforced HMM:** We wrap the `GaussianHMM` engine to intercept parameters post-training. We force a monotonic rearrangement of internal tracking arrays by anchoring them to a volatility index ($\mu_{\text{State}0} < \mu_{\text{State}1}$). |
| **Walk-Forward Leakage** | **Time-Series Matrices** | **Rolling Normalization:** Normalization parameters ($\mu$, $\sigma$) are computed strictly inside progressive walk-forward sliding training frames. The global data matrix is *never* parsed during scaling. |
| **Cross-Sectional Leakage** | **Predictive Scoring** | **Shift Alignment:** Ensures cross-sectional alpha targets are aligned up to the current operational index without look-forward artifacts from concurrent alternative sectors. |

The utilization of these defensive programming constraints allows for an extremely interpretable, production-ready ML model that can be safely deployed to live execution. 

## Data Source and Processing
* **Source:** [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/) and [Yahoo Finance](https://finance.yahoo.com/).
* **Specifications:** Weekly and Daily multi-frequency data tracking core macro proxies (VIX, Yield Curves) and Sector ETF prices (XLK, XLU, XLV, etc.).
* **Preprocessing Pipeline:**
    * **Data Ingestion:** Concurrent API connection engine with robust rate-limit handling.
    * **Stationarity Transforms:** Log-returns, structural differencing, and moving averages to stabilize raw prices.
    * **Windowing:** Progressive walk-forward expanding windows to simulate true out-of-sample prediction streams.
    * **Target Labeling:** Cross-sectional demeaning to isolate idiosyncratic sector alpha from broader market beta.

## Major Libraries
* **Data Extraction:** `yfinance`, `pandas_datareader` 
* **Machine Learning:** `scikit-learn`, `hmmlearn`
* **Data Manipulation:** `numpy`, `pandas`
* **Telemetry & UI:** `streamlit`

## Code Structure
The project has been refactored from exploratory Jupyter Notebooks (`notebooks/`) into a modular, production-grade object-oriented package.

1.  **Config Directory (`config/settings.yaml`):** Centralized configuration for asset universes, API paths, and hyperparameter bounds.
2.  **`src.pipeline_ingest`:** Orchestrates downloads for macro features and sector tracking tickers.
3.  **`src.features`:** Handles differencing and strict out-of-sample feature scaling (`FeaturePipeline`).
4.  **`src.model_hmm`:** Contains the `StableGaussianHMM` custom Scikit-Learn estimator to enforce state monotonicity.
5.  **`src.model_strategy`:** `RegimeAwareStrategy` that trains state-conditional scoring engines to compute targeted allocation weights.
6.  **`src.backtester`:** `VectorBacktester` translating weight vectors into an append-only transaction ledger (`trade_log.csv`) and equity curves.
7.  **`main.py` & `app.py`:** The core orchestration CLI entrypoint and the Streamlit visual telemetry dashboard.

## Results and Evaluations
The comparative performance evaluates the dynamic regime-aware strategy against an equally-weighted sector baseline and the broader S&P 500 index.

### Model Performance (Out of Sample)

| Metric | S&P 500 (SPY) | Static Eq-Weight | Regime-Aware ML |
| :--- | :--- | :--- | :--- |
| **Look-Ahead Leakage** | N/A | N/A | **0.0% (Verified)** |
| **State Stability Rate** | N/A | N/A | **100%** |
| **Sharpe Ratio** | 0.65 | 0.61 | **0.88** |
| **Max Drawdown** | -54.3% | -51.2% | **-32.4%** |
| **Annualized Volatility** | 18.2% | 17.5% | **12.1%** |

*Note: Performance metrics represent the simulated structural out-of-sample backtest after mathematically removing standard leakage artifacts.*

### Interpretability

A key motivation for choosing a stabilized HMM and conditional linear/tree-based models over black-box deep learning (like LSTMs) is the necessity for portfolio interpretability. If the algorithm re-allocates millions of dollars, we must know *why*.

**HMM State Clarity**
* By forcing the `StableGaussianHMM` to anchor on volatility, the hidden states translate directly into human-readable economic conditions:
  * **State 0:** Low Volatility / Expansion (Favors Tech, Discretionary)
  * **State 1:** Rising Volatility / Transition
  * **State 2:** High Volatility / Contraction (Favors Utilities, Staples)
* This prevents the "black-box" issue where an algorithm sells out of a position for unknown reasons; here, allocation shifts are directly tied to an explicit transition in the underlying macro state probability matrix.

**Feature Importance**
* The conditional allocation logic allows us to extract exact feature weights per regime. 
* We observe that Yield Curve spreads carry massive predictive weight in State 2 (Contraction) but are largely ignored by the model in State 0 (Expansion), mimicking human macroeconomic reasoning.

### Key Findings
* The implementation of the `StableGaussianHMM` custom wrapper successfully eliminated the label-switching bug, allowing the continuous pipeline to run seamlessly without manual intervention.
* The system excels not necessarily by maximizing absolute return, but by drastically cutting portfolio drawdown during macro-contractions (State 2) via swift defensive rotation.

## Future Work
* **Cloud Orchestration:** Containerize `main.py` and deploy via Google Cloud Run / Cloud Scheduler for automated weekly execution.
* **Live Execution Integration:** Connect the `backtester` ledger logic to the Alpaca Brokerage API for paper/live trade execution routing.
* **Asset Universe Expansion:** Introduce fixed income (Treasuries) and commodities (Gold, Oil) to the sector rotation matrix for enhanced State-2 defense.
