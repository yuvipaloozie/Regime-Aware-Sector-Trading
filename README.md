# Regime Based Sector Rotation Using HMMs and XGBRanker

An end-to-end automated data pipeline and predictive modeling framework that identifies macroeconomic macro-states (regimes) and dynamically updates structural allocation across economic sectors. 

This project bridges data engineering, unsupervised state modeling, and predictive analytics to simulate an enterprise-grade automated execution engine.

---

## The Systemic Engineering Challenge

Many machine learning applications fail in dynamic environments because underlying data distributions shift unexpectedly. In manufacturing plants, a distillation column shifts behaviors based on throughput states; similarly, financial markets shift baseline behaviors based on macroeconomic conditions. 

### Core Engineering Inversions Solved:
1. **Dynamic State Destabilization (Label-Switching):** Unsupervised models like Hidden Markov Models (HMMs) classify conditions without a predefined scale. Across different training windows, "State 0" might arbitrarily flip from representing "Low Volatility" to "High Volatility," breaking downstream scoring logic.
2. **Temporal Feature Leakage:** Standard global feature scaling injects future parameters (mean/variance) into historical training arrays. This framework strictly partitions rolling window parameters to guarantee out-of-sample mathematical validity.
Technical Component Details
Ingestion Engine (src/pipeline_ingest.py)
Provides concurrent API connectors to capture historical data arrays. It isolates web errors, manages rate limits, and structures raw multi-frequency files down into clean local cache tensors.

DataIngestor: Orchestrates downloads for the designated macro features and sector tracking tickers outlined in the configuration layers.
---
