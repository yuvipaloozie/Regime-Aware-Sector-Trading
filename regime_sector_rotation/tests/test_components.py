import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline_ingest import load_config
from src.features import engineer_hmm_features
from src.model_hmm import AnchoredGaussianHMM
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
