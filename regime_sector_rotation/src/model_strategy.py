import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm

def run_walk_forward_xgboost(df_master, config):
    """
    Fits rolling XGBRanker models to predict sector forward alpha.
    Zero-leakage is preserved via purging 4-week forward overlap.
    """
    xgb_cfg = config['xgboost']
    features = [f"{col}_Z" for col in ['Mom_1W', 'Mom_13W', 'Vol_13W', 'Beta_SPY_26W', 'Corr_TNX_26W']] + ['Smoothed_Regime']
    
    # 1. Prep trainable data and sort multi-index
    df_trainable = df_master[df_master['Is_Training_Row'] == True].copy()
    df_trainable['Smoothed_Regime'] = df_trainable['Smoothed_Regime'].astype(int)
    df_trainable = df_trainable.sort_index()
    
    dates = df_trainable.index.get_level_values('Date').unique().sort_values()
    
    # 2. Extract walk-forward hyperparameters
    min_train = xgb_cfg['min_train_weeks']
    purge = xgb_cfg['purge_weeks']
    step = xgb_cfg['step_weeks']
    
    xgb_params = {
        'objective': 'rank:pairwise',
        'learning_rate': xgb_cfg['learning_rate'],
        'max_depth': xgb_cfg['max_depth'],
        'n_estimators': xgb_cfg['n_estimators'],
        'subsample': xgb_cfg['subsample'],
        'colsample_bytree': xgb_cfg['colsample_bytree'],
        'random_state': xgb_cfg['random_state'],
        'n_jobs': -1
    }
    
    predictions = []
    feature_importance_history = []
    
    print(f"Running expanding window walk-forward XGBoost on {len(dates)} dates...")
    
    for t_idx in tqdm(range(min_train + purge, len(dates), step), desc="XGBoost Folds"):
        # Define historical training cutoff and test range
        train_cutoff_date = dates[t_idx - purge]
        test_start_date = dates[t_idx]
        test_end_date = dates[min(t_idx + step - 1, len(dates) - 1)]
        
        train_data = df_trainable.loc[:train_cutoff_date]
        test_data = df_trainable.loc[test_start_date:test_end_date]
        
        if test_data.empty:
            continue
            
        X_train = train_data[features]
        y_train = train_data['Target_Alpha_4W']
        
        # XGBRanker requires 'group' specifying queries (sectors active each week)
        train_groups = train_data.groupby(level='Date').size().values
        
        # 3. Fit XGBRanker
        model = xgb.XGBRanker(**xgb_params)
        model.fit(X_train, y_train, group=train_groups, verbose=False)
        
        # Track feature importance
        feature_importance_history.append(model.feature_importances_)
        
        # 4. Predict out-of-sample
        X_test = test_data[features]
        test_preds = test_data[['Raw_Fwd_4W', 'Smoothed_Regime', 'Risk_Scalar']].copy()
        test_preds['ML_Rank_Score'] = model.predict(X_test)
        
        predictions.append(test_preds)
        
    if len(predictions) == 0:
        raise ValueError("No predictions generated. Check date/length configuration.")
        
    oos_results = pd.concat(predictions)
    
    # Store feature importance stats for reporting
    avg_importances = np.mean(feature_importance_history, axis=0)
    feat_importance = dict(zip(features, avg_importances))
    
    print(f"XGBoost walk-forward complete. Generated {len(oos_results)} prediction rows.")
    return oos_results, feat_importance

def calculate_portfolio_weights(oos_results, df_master, config):
    """
    Simulates capital allocation using dynamic hysteresis:
      - Top N sectors are chosen.
      - An asset is only sold if its rank falls below (Top N + Buffer).
      - Under high stress (from HMM Risk_Scalar), allocation scales out of sectors into TLT.
    """
    portfolio_cfg = config['portfolio']
    top_n = portfolio_cfg['top_n']
    buffer = portfolio_cfg['buffer']
    
    dates = oos_results.index.get_level_values('Date').unique().sort_values()
    tickers = oos_results.index.get_level_values('Ticker').unique()
    
    # Rebalance every 4 weeks as per target window
    rebalance_dates = dates[::4]
    
    matrix_cols = list(tickers) + ['TLT']
    weight_matrix = pd.DataFrame(0.0, index=rebalance_dates, columns=matrix_cols)
    
    current_holdings = []
    
    for date in rebalance_dates:
        if date not in oos_results.index:
            continue
            
        day_data = oos_results.xs(date, level='Date').copy()
        day_data['Rank'] = day_data['ML_Rank_Score'].rank(ascending=False)
        risk_scalar = day_data['Risk_Scalar'].iloc[0] if not day_data.empty else 0.0
        
        # Hysteresis Portfolio Selection
        # Keep old holding if it stays within (top_n + buffer) rank
        retained_holdings = [
            t for t in current_holdings 
            if t in day_data.index and day_data.loc[t, 'Rank'] <= (top_n + buffer)
        ]
        
        slots_to_fill = top_n - len(retained_holdings)
        available_buys = day_data[~day_data.index.isin(retained_holdings)].sort_values('Rank')
        new_buys = available_buys.head(slots_to_fill).index.tolist()
        
        current_holdings = retained_holdings + new_buys
        
        # Allocate weights based on Risk_Scalar
        if risk_scalar > 0.0 and len(current_holdings) > 0:
            equity_weight = (1.0 / len(current_holdings)) * risk_scalar
            weight_matrix.loc[date, current_holdings] = equity_weight
            
        tlt_weight = 1.0 - risk_scalar
        if tlt_weight > 0.0:
            weight_matrix.loc[date, 'TLT'] = tlt_weight
            
    return weight_matrix

def backtest_strategy(weight_matrix, df_master, df_weekly, config):
    """
    Computes trading turnover, slippage friction, and net returns.
    Includes benchmark (Equal-Weight active sectors) comparison.
    """
    portfolio_cfg = config['portfolio']
    slippage_bps = portfolio_cfg['slippage_bps']
    risk_free_rate = portfolio_cfg['risk_free_rate_weekly']
    
    rebalance_dates = weight_matrix.index
    
    # 1. Prevent Lookahead Bias: Shift target weights by 1 period (next 4-week holding period)
    target_weights = weight_matrix.shift(1).fillna(0.0)
    
    # Calculate turnover (absolute changes in weight vector)
    turnover = target_weights.diff().abs().sum(axis=1)
    
    # 2. Extract 4-week forward return matrix
    return_matrix = df_master['Fwd_Ret_4W'].unstack().reindex(index=rebalance_dates).fillna(0.0)
    
    # Calculate TLT forward return over 4-week block
    tlt_fwd_ret_4w = df_weekly['TLT'].shift(-4) / df_weekly['TLT'] - 1
    return_matrix['TLT'] = tlt_fwd_ret_4w.reindex(rebalance_dates).fillna(0.0)
    
    # 3. Calculate gross and net returns
    gross_returns = (target_weights * return_matrix).sum(axis=1)
    net_returns = gross_returns - (turnover * slippage_bps)
    
    # 4. Benchmarks (11 Sector Equal-Weighted return)
    benchmark_returns = df_master['Fwd_Ret_4W'].unstack().reindex(index=rebalance_dates).fillna(0.0).mean(axis=1)
    
    # 5. Cumulative wealth paths
    cum_strategy = (1 + net_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()
    
    # 6. Summary metrics
    excess_returns = net_returns - risk_free_rate
    # 13 4-week periods in a 52-week year
    sharpe_ratio = np.sqrt(13) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0.0
    
    metrics = {
        'Ann. Sharpe': sharpe_ratio,
        'Strategy Total Return (%)': (cum_strategy.iloc[-1] - 1) * 100 if not cum_strategy.empty else 0.0,
        'Benchmark Total Return (%)': (cum_benchmark.iloc[-1] - 1) * 100 if not cum_benchmark.empty else 0.0,
        'Max Drawdown (%)': ((cum_strategy - cum_strategy.cummax()) / cum_strategy.cummax()).min() * 100 if not cum_strategy.empty else 0.0,
        'Avg 4Wk Turnover (%)': turnover.mean() * 100
    }
    
    return {
        'net_returns': net_returns,
        'benchmark_returns': benchmark_returns,
        'cum_strategy': cum_strategy,
        'cum_benchmark': cum_benchmark,
        'turnover': turnover,
        'metrics': metrics
    }
