import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure the project root is in the Python search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline_ingest import DataPipeline
from src.features import engineer_hmm_features, engineer_sector_features
from src.model_hmm import run_walk_forward_hmm
from src.model_strategy import (
    run_walk_forward_xgboost,
    evaluate_rank_predictions,
    calculate_portfolio_weights,
    backtest_strategy
)
from src.backtester import generate_quantstats_tearsheet

class SimulatedTradeManager:
    """
    Chronological trade manager that simulates portfolio execution.
    Logs transactions, weight allocations, and equity values to a unified CSV.
    """
    def __init__(self, initial_cash=100000.0, log_path="data/trade_log.csv", transaction_cost_bps=0.0,
                 cash_symbol="CASH"):
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.holdings = {}  # Asset -> Shares
        self.log_path = log_path
        self.transaction_cost_rate = transaction_cost_bps / 10000.0
        self.cash_symbol = cash_symbol
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Write headers if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                f.write("Timestamp,Asset,Action,Vol,CostBasis\n")

    def execute_trades(self, timestamp, target_weights, prices, spy_price, spy_start_price, initial_cash=100000.0, regime=None, risk_scalar=None):
        """
        Calculates and executes trades for a specific timestamp.
        Logs transactions, allocations, and equity values to data/trade_log.csv.
        """
        # 1. Update Portfolio Value based on current prices
        current_holdings_value = 0.0
        for asset, shares in self.holdings.items():
            if asset in prices:
                current_holdings_value += shares * prices[asset]
                
        self.portfolio_value = self.cash + current_holdings_value
        
        # 2. Update SPY Benchmark Equity
        spy_benchmark_val = initial_cash * (spy_price / spy_start_price)
        
        # 3. Calculate target shares and execute trades
        new_holdings = {}
        trade_logs = []
        
        for asset, weight in target_weights.items():
            if weight <= 0.0:
                continue
            if asset == self.cash_symbol:
                continue
                
            price = prices.get(asset)
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(f"Missing or invalid execution price for {asset} at {timestamp}")
            target_value = weight * self.portfolio_value
            target_shares = round(target_value / price, 2)
            
            current_shares = self.holdings.get(asset, 0.0)
            diff_shares = round(target_shares - current_shares, 2)
            
            if abs(diff_shares) >= 0.01:
                action = "BUY" if diff_shares > 0 else "SELL"
                trade_logs.append({
                    "Timestamp": timestamp,
                    "Asset": asset,
                    "Action": action,
                    "Vol": abs(diff_shares),
                    "CostBasis": price
                })
                notional = abs(diff_shares * price)
                self.cash -= diff_shares * price
                self.cash -= notional * self.transaction_cost_rate
                
            new_holdings[asset] = target_shares
            
        # Sell off any asset no longer in the target weights
        for asset, shares in list(self.holdings.items()):
            if asset not in target_weights or target_weights[asset] <= 0.0:
                price = prices.get(asset)
                if price is None or not np.isfinite(price) or price <= 0:
                    raise ValueError(f"Missing or invalid liquidation price for {asset} at {timestamp}")
                trade_logs.append({
                    "Timestamp": timestamp,
                    "Asset": asset,
                    "Action": "SELL",
                    "Vol": shares,
                    "CostBasis": price
                })
                self.cash += shares * price
                self.cash -= abs(shares * price) * self.transaction_cost_rate
                
        self.holdings = new_holdings
        
        # 4. Write records to CSV log
        with open(self.log_path, 'a') as f:
            # Write actual trade transactions
            for log in trade_logs:
                f.write(f"{log['Timestamp']},{log['Asset']},{log['Action']},{log['Vol']:.2f},{log['CostBasis']:.2f}\n")
                
            # Write allocation weights for visual tracking
            for asset, weight in target_weights.items():
                f.write(f"{timestamp},{asset},ALLOCATION,{weight:.4f},0.00\n")
                
            # Write overall portfolio equity metrics
            f.write(f"{timestamp},PORTFOLIO_METRIC,EQUITY,{self.portfolio_value:.2f},{spy_benchmark_val:.2f}\n")
            
            # Write regime metrics if available
            if regime is not None and risk_scalar is not None:
                f.write(f"{timestamp},REGIME_METRIC,STATE,{regime},{risk_scalar:.4f}\n")

def execute_orchestration(weeks=26, force_refresh=False, data_mode=None):
    print("=" * 70)
    print("      REGIME-CONDITIONED SECTOR ROTATION PRODUCTION ENGINE")
    print("=" * 70)
    
    # 1. Initialize Ingestion Pipeline
    pipeline = DataPipeline(data_mode=data_mode)
    config = pipeline.config
    
    # Check if cache files should be removed
    if force_refresh:
        for f in ["yfinance_adjusted.csv", "fred_point_in_time.csv"]:
            p = os.path.join(pipeline.raw_dir, f)
            if os.path.exists(p):
                os.remove(p)
            meta = f"{p}.meta.json"
            if os.path.exists(meta):
                os.remove(meta)
                
    # 2. Ingest and Resample Data
    df_weekly = pipeline.get_processed_data()
    print(f"Data resampled successfully. Date Range: {df_weekly.index.min().date()} to {df_weekly.index.max().date()}")
    
    # 3. Create HMM Features
    print("\nEngineering HMM market stress features...")
    df_hmm_features = engineer_hmm_features(df_weekly)
    
    # 4. Fit Walk-Forward HMM Classifier
    print("\nExecuting State-Stabilized Walk-Forward HMM Classification...")
    df_hmm_oos = run_walk_forward_hmm(df_hmm_features, config)
    
    # 5. Create Sector Features & Join smoothed macro regimes
    print("\nEngineering robust sector-rotation features & targets...")
    df_sector_weekly = engineer_sector_features(df_weekly, config)
    
    # Join the out-of-sample regime predictions
    df_master = df_sector_weekly.join(df_hmm_oos, how='inner')
    
    # 6. Fit Walk-Forward XGBRanker Model
    print("\nRunning Expanding Window Walk-Forward XGBRanker...")
    oos_results, feat_importance = run_walk_forward_xgboost(df_master, config)
    rank_metrics = evaluate_rank_predictions(oos_results, config['portfolio']['top_n'])
    
    # Display feature importance
    print("\nGlobal Feature Importance (Information Gain):")
    for feat, val in sorted(feat_importance.items(), key=lambda x: x[1], reverse=True):
        print(f" - {feat:<20}: {val:.4f}")
        
    # 7. Portfolio Allocations calculation
    print("\nCalculating dynamic hysteresis portfolio weights...")
    weight_matrix = calculate_portfolio_weights(oos_results, df_master, config, df_weekly=df_weekly)
    
    # 8. Standard Backtest & QuantStats Tearsheet Generation
    print("\nRunning strategy backtest & performance metrics verification...")
    backtest = backtest_strategy(
        weight_matrix, df_master, df_weekly, config, df_prices_daily=pipeline.daily_equity
    )
    
    metrics = backtest['metrics']
    print("\n" + "=" * 50)
    print("           BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Strategy Total Return:      {metrics['Strategy Total Return (%)']:.2f}%")
    print(f"Benchmark Total Return:     {metrics['Benchmark Total Return (%)']:.2f}%")
    print(f"Ann. Sharpe Ratio:          {metrics['Ann. Sharpe']:.2f}")
    print(f"Max Drawdown:               {metrics['Max Drawdown (%)']:.2f}%")
    print(f"Strategy CAGR:              {metrics['Strategy CAGR (%)']:.2f}%")
    print(f"Avg Annual Turnover:        {metrics['Avg Annual Turnover (%)']:.2f}%")
    print(f"Mean Rank IC:               {rank_metrics['Mean Rank IC']:.3f}")
    print(f"Top-{config['portfolio']['top_n']} Hit Rate:            {rank_metrics['Top-N Hit Rate']:.1%}")
    print("=" * 50)
    
    # Export tearsheet
    generate_quantstats_tearsheet(backtest['net_returns'], backtest['benchmark_returns'])
    
    # Save the entire backtest cumulative equity path (Strategy vs SPY benchmark)
    df_backtest_equity = pd.DataFrame({
        'Strategy': backtest['cum_strategy'],
        'Benchmark': backtest['cum_benchmark'],
        'SPY': backtest['cum_spy'],
        'Momentum': backtest['cum_momentum'],
    })
    df_backtest_equity.index.name = 'Date'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    df_backtest_equity.to_csv(os.path.join(data_dir, "backtest_equity.csv"))
    
    # Save the entire backtest sector weightings history
    weight_matrix.to_csv(os.path.join(data_dir, "backtest_weights.csv"))

    pipeline.write_run_provenance(
        os.path.join(data_dir, "run_manifest.json"),
        extra={
            "metrics": metrics,
            "rank_metrics": rank_metrics,
            "cost_sensitivity": backtest['cost_sensitivity'],
            "feature_importance": feat_importance,
            "signal_start": str(weight_matrix.index.min()),
            "signal_end": str(weight_matrix.index.max()),
            "execution_start": str(backtest['execution_schedule'].index.min()),
            "execution_end": str(backtest['execution_schedule'].index.max()),
        },
    )
    
    # 9. Simulated Trade Manager Simulation
    # Run a chronological weekly simulation over the requested final weeks
    execution_schedule = backtest['execution_schedule']
    execution_signal_dates = backtest['execution_signal_dates']
    sim_weeks = min(weeks, len(execution_schedule))
    print(f"\nInitializing simulated trade manager for the last {sim_weeks} rebalance periods...")
    
    # Get active prices for trade execution (close prices)
    tradable_columns = [column for column in execution_schedule.columns if column != config['portfolio']['cash_symbol']]
    prices_df = pipeline.daily_equity[tradable_columns]
    spy_df = pipeline.daily_equity[config['benchmarks']['equity']]
    
    # Select dates to simulate
    sim_dates = execution_schedule.index[-sim_weeks:]
    spy_start_price = spy_df.loc[sim_dates[0]]
    
    # Clean previous logs to keep it a fresh walkthrough
    log_path = os.path.join(data_dir, "trade_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)
        
    trade_manager = SimulatedTradeManager(
        initial_cash=config['portfolio']['initial_cash'],
        log_path=log_path,
        transaction_cost_bps=config['portfolio']['transaction_cost_bps'],
        cash_symbol=config['portfolio']['cash_symbol'],
    )
    
    for idx, date in enumerate(sim_dates):
        date_str = str(date.date())
        # Target weights for this week
        target_w = execution_schedule.loc[date].to_dict()
        # Active execution close prices
        prices_t = prices_df.loc[date].to_dict()
        spy_price_t = spy_df.loc[date]
        
        # Get regime and risk scalar for this date
        signal_date = execution_signal_dates.loc[date]
        day_master = df_master.xs(signal_date, level='Date')
        regime_val = int(day_master['Smoothed_Regime'].iloc[0])
        risk_scalar_val = float(day_master['Risk_Scalar'].iloc[0])
        
        trade_manager.execute_trades(
            timestamp=date_str,
            target_weights=target_w,
            prices=prices_t,
            spy_price=spy_price_t,
            spy_start_price=spy_start_price,
            initial_cash=config['portfolio']['initial_cash'],
            regime=regime_val,
            risk_scalar=risk_scalar_val
        )
        
    print(f"SUCCESS: Simulated trade log successfully populated at: {log_path}")
    print("Orchestration pipeline run finished successfully.")

def run_pipeline():
    parser = argparse.ArgumentParser(description="Unified Orchestration Entrypoint")
    parser.add_argument("--weeks", type=int, default=26, help="Number of final weeks to run simulation on")
    parser.add_argument("--force-refresh", action="store_true", help="Force download and ignore caches")
    parser.add_argument("--demo", action="store_true", help="Explicitly use deterministic synthetic demo data")
    args = parser.parse_args()
    
    execute_orchestration(weeks=args.weeks, force_refresh=args.force_refresh, data_mode="demo" if args.demo else None)

if __name__ == "__main__":
    run_pipeline()
