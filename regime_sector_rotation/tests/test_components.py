import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_ingest import DataPipeline, load_config, splice_proxy_returns
from src.features import engineer_hmm_features
from src.model_hmm import AnchoredGaussianHMM, causal_filter_probabilities
from src.model_strategy import (
    build_execution_schedule,
    backtest_strategy,
    iter_purged_walk_forward_splits,
    run_walk_forward_xgboost,
)
from main import SimulatedTradeManager

class TestRegimeSectorRotation(unittest.TestCase):
    def setUp(self):
        # Create a simple mock configuration dictionary
        self.config = {
            'sectors': ['XLK', 'XLE', 'XLF'],
            'proxies': {},
            'benchmarks': {
                'equity': 'SPY',
                'volatility': '^VIX',
                'yield_10y': '^TNX',
                'safe_haven': 'TLT'
            },
            'portfolio': {
                'initial_cash': 100000.0
            }
        }

    def test_config_loading(self):
        """Verify settings.yaml configuration loads successfully."""
        try:
            config = load_config()
            self.assertIn('sectors', config)
            self.assertIn('benchmarks', config)
            self.assertIn('hmm', config)
            self.assertIn('xgboost', config)
        except Exception as e:
            self.fail(f"Config load failed: {e}")

    def test_anchored_hmm_sorting(self):
        """Test that AnchoredGaussianHMM correctly sorts states by VIX mean ascending."""
        np.random.seed(42)
        
        # Generate dummy 2D data: 3 components, dimension 2 (second dimension is VIX)
        # Component 0: VIX mean ~ 12
        # Component 1: VIX mean ~ 35
        # Component 2: VIX mean ~ 22
        c0 = np.random.normal(loc=[0.01, 12.0], scale=[0.005, 1.0], size=(100, 2))
        c1 = np.random.normal(loc=[-0.02, 35.0], scale=[0.01, 3.0], size=(100, 2))
        c2 = np.random.normal(loc=[-0.005, 22.0], scale=[0.007, 1.5], size=(100, 2))
        
        X = np.vstack([c0, c1, c2])
        
        # Fit Anchored HMM (VIX is index 1)
        model = AnchoredGaussianHMM(n_components=3, vix_idx=1, random_state=42)
        model.fit(X)
        
        # Verify the fitted model means are sorted ascending by VIX level
        fitted_means = model.model_.means_[:, 1]
        self.assertTrue(np.all(np.diff(fitted_means) >= 0.0), f"Fitted VIX means are not sorted: {fitted_means}")

    def test_hmm_filter_is_causal(self):
        """Changing a later test observation must not change an earlier state posterior."""
        rng = np.random.default_rng(7)
        history = np.vstack([rng.normal([0, 12], [0.5, 1], (100, 2)),
                             rng.normal([0, 30], [0.5, 2], (100, 2))])
        model = AnchoredGaussianHMM(n_components=2, vix_idx=1, random_state=4).fit(history)
        first = np.array([[0.0, 18.0]])
        ordinary_future = np.array([[0.0, 19.0]])
        extreme_future = np.array([[5.0, 100.0]])
        ordinary = causal_filter_probabilities(model, history, np.vstack([first, ordinary_future]))
        extreme = causal_filter_probabilities(model, history, np.vstack([first, extreme_future]))
        np.testing.assert_allclose(ordinary[0], extreme[0], atol=1e-12)

    def test_purged_split_excludes_label_boundary(self):
        dates = pd.date_range("2020-01-03", periods=12, freq="W-FRI")
        train, test = next(iter(iter_purged_walk_forward_splits(dates, min_train=5, purge=2, step=2)))
        self.assertEqual(list(train), list(dates[:5]))
        self.assertEqual(list(test), list(dates[7:9]))
        self.assertLess(train[-1], dates[5])

    def test_proxy_splice_preserves_switch_continuity(self):
        idx = pd.date_range("2020-01-01", periods=6)
        target = pd.Series([np.nan, np.nan, np.nan, 100.0, 102.0, 101.0], index=idx)
        proxy = pd.Series([40.0, 42.0, 44.0, 46.0, 47.0, 48.0], index=idx)
        result = splice_proxy_returns(target, proxy)
        self.assertEqual(result.loc[idx[3]], 100.0)
        self.assertAlmostEqual(result.loc[idx[2]], 100.0 * 44.0 / 46.0)

    def test_synthetic_generation_requires_explicit_demo_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = DataPipeline(raw_dir=temp_dir, data_mode="real")
            with self.assertRaises(RuntimeError):
                pipeline.generate_synthetic_equity_data(["SPY"], "2020-01-01", "2020-02-01")

    def test_execution_schedule_is_after_signal(self):
        signals = pd.DatetimeIndex(["2024-01-05", "2024-01-12"])
        weights = pd.DataFrame({"XLK": [1.0, 0.0], "CASH": [0.0, 1.0]}, index=signals)
        market_days = pd.bdate_range("2024-01-01", "2024-01-19")
        schedule, signal_map = build_execution_schedule(weights, market_days, lag_trading_days=1)
        self.assertTrue((schedule.index > signal_map.values).all())
        self.assertEqual(schedule.index[0], pd.Timestamp("2024-01-08"))

    def test_ranker_scores_latest_unlabeled_rows(self):
        dates = pd.date_range("2022-01-07", periods=14, freq="W-FRI")
        index = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["Date", "Ticker"])
        rng = np.random.default_rng(12)
        frame = pd.DataFrame(index=index)
        for feature in ["Mom_1W", "Mom_13W", "Vol_13W", "Beta_SPY_26W", "Corr_TNX_26W"]:
            frame[f"{feature}_Z"] = rng.normal(size=len(frame))
        frame["Regime_Prob_0"] = 0.6
        frame["Regime_Prob_1"] = 0.4
        frame["Smoothed_Regime"] = 0
        frame["Risk_Scalar"] = 0.9
        frame["Raw_Fwd_4W"] = rng.normal(size=len(frame))
        frame["Target_Alpha_4W"] = frame.groupby(level="Date")["Raw_Fwd_4W"].transform(lambda values: values - values.mean())
        frame.loc[(dates[-2:], slice(None)), ["Raw_Fwd_4W", "Target_Alpha_4W"]] = np.nan
        frame["Is_Training_Row"] = frame["Target_Alpha_4W"].notna()
        config = {"xgboost": {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 5,
                               "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 1,
                               "purge_weeks": 2, "step_weeks": 2, "min_train_weeks": 5}}
        results, _ = run_walk_forward_xgboost(frame, config)
        self.assertIn(dates[-1], results.index.get_level_values("Date"))
        self.assertTrue(results.xs(dates[-1], level="Date")["Target_Alpha_4W"].isna().all())

    def test_backtest_uses_common_post_execution_window(self):
        market_days = pd.bdate_range("2024-01-01", periods=12)
        prices = pd.DataFrame({"A": np.linspace(100, 111, 12), "SPY": np.linspace(200, 211, 12)}, index=market_days)
        signals = pd.DataFrame({"A": [1.0], "CASH": [0.0]}, index=[pd.Timestamp("2024-01-05")])
        config = {"sectors": ["A"], "benchmarks": {"equity": "SPY"},
                  "portfolio": {"cash_symbol": "CASH", "annual_risk_free_rate": 0.0,
                                "execution_lag_trading_days": 1, "transaction_cost_bps": 0.0}}
        result = backtest_strategy(signals, pd.DataFrame(), prices, config, df_prices_daily=prices)
        self.assertEqual(result["net_returns"].index[0], pd.Timestamp("2024-01-08"))
        self.assertEqual(result["benchmark_returns"].index[0], result["net_returns"].index[0])

    def test_hmm_feature_engineering(self):
        """Verify HMM stress factor calculations on mock daily/weekly series."""
        idx = pd.date_range(start="2020-01-01", periods=10, freq="W-FRI")
        df_mock = pd.DataFrame(index=idx)
        df_mock['SPY'] = np.linspace(300, 310, 10)
        df_mock['VIX'] = np.linspace(15, 20, 10)
        df_mock['T10Y2Y'] = np.linspace(1.2, 0.8, 10)
        df_mock['STLFSI4'] = np.linspace(-0.5, 0.2, 10)
        df_mock['CPIAUCSL'] = np.linspace(250, 252, 10)
        
        df_feats = engineer_hmm_features(df_mock)
        
        expected_cols = ['SPY_Log_Ret_W', 'VIX_Level', 'Yield_Curve_Change_W', 'Financial_Stress', 'Inflation_MoM']
        for col in expected_cols:
            self.assertIn(col, df_feats.columns)
        self.assertFalse(df_feats.empty)

    def test_simulated_trade_manager(self):
        """Verify SimulatedTradeManager chronologically handles transactions and equity paths."""
        log_path = "data/test_trade_log.csv"
        if os.path.exists(log_path):
            os.remove(log_path)
            
        trade_manager = SimulatedTradeManager(initial_cash=100000.0, log_path=log_path)
        
        # Define simulation inputs
        timestamp = "2026-05-22"
        target_weights = {"XLK": 0.50, "XLE": 0.50}
        prices = {"XLK": 100.0, "XLE": 50.0, "TLT": 90.0}
        spy_price = 400.0
        spy_start_price = 400.0
        
        # Execute Trades
        trade_manager.execute_trades(
            timestamp=timestamp,
            target_weights=target_weights,
            prices=prices,
            spy_price=spy_price,
            spy_start_price=spy_start_price
        )
        
        # Check holdings were created
        self.assertEqual(trade_manager.holdings.get("XLK"), 500.0) # 50,000 / 100
        self.assertEqual(trade_manager.holdings.get("XLE"), 1000.0) # 50,000 / 50
        
        # Validate that the test log CSV exists and has appropriate rows
        self.assertTrue(os.path.exists(log_path))
        df_log = pd.read_csv(log_path)
        self.assertFalse(df_log.empty)
        
        # Clean up test log
        if os.path.exists(log_path):
            os.remove(log_path)

if __name__ == "__main__":
    unittest.main()
