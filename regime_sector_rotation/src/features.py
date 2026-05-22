import numpy as np
import pandas as pd

def engineer_hmm_features(df_weekly):
    """
    Computes features for the HMM market regime classifier.
    
    Expected columns in df_weekly:
        - SPY
        - VIX
        - T10Y2Y
        - STLFSI4
        - CPIAUCSL
    """
    df_features = pd.DataFrame(index=df_weekly.index)
    
    # 1. Weekly Log Returns of SPY
    df_features['SPY_Log_Ret_W'] = np.log(df_weekly['SPY'] / df_weekly['SPY'].shift(1))
    
    # 2. Implied Risk & Stress (Levels / Weekly Differences)
    df_features['VIX_Level'] = df_weekly['VIX']
    df_features['Yield_Curve_Change_W'] = df_weekly['T10Y2Y'].diff(1)
    df_features['Financial_Stress'] = df_weekly['STLFSI4']
    
    # 3. Inflation Proxy (Month-over-Month CPI Growth, 4-week lookback)
    df_features['Inflation_MoM'] = np.log(df_weekly['CPIAUCSL'] / df_weekly['CPIAUCSL'].shift(4))
    
    df_features = df_features.dropna()
    return df_features

def engineer_sector_features(df_weekly, config):
    """
    Computes multi-dimensional features for downstream XGBRanker.
    Combines momentum, volatility, beta, and rate sensitivity, then Z-scores cross-sectionally.
    """
    sectors = config['sectors']
    spy_ticker = config['benchmarks']['equity']
    tnx_ticker = config['benchmarks']['yield_10y']
    
    # 1. Extract raw series
    weekly_etfs = df_weekly[sectors]
    weekly_spy = df_weekly[spy_ticker]
    weekly_tnx = df_weekly[tnx_ticker]
    
    # 2. Calculate base returns and differences
    weekly_rets = np.log(weekly_etfs / weekly_etfs.shift(1))
    spy_rets = np.log(weekly_spy / weekly_spy.shift(1))
    tnx_diff = weekly_tnx.diff(1)
    
    # 3. Build features
    # A. Momentum (1-Week Reversion, 13-Week Trend)
    mom_1w = weekly_rets
    mom_13w = np.log(weekly_etfs / weekly_etfs.shift(13))
    
    # B. Volatility (13-Week Annualized)
    vol_13w = weekly_rets.rolling(13).std() * np.sqrt(52)
    
    # C. Market Beta (26-Week Rolling)
    spy_var_26w = spy_rets.rolling(26).var()
    beta_26w = pd.DataFrame(index=weekly_rets.index, columns=weekly_rets.columns)
    for col in weekly_rets.columns:
        beta_26w[col] = weekly_rets[col].rolling(26).cov(spy_rets) / spy_var_26w
        
    # D. Rate Sensitivity (26-Week Rolling Correlation to 10Yr Yield)
    corr_tnx_26w = pd.DataFrame(index=weekly_rets.index, columns=weekly_rets.columns)
    for col in weekly_rets.columns:
        corr_tnx_26w[col] = weekly_rets[col].rolling(26).corr(tnx_diff)
        
    # 4. Stack features
    def stack_feature(df, feature_name):
        stacked = df.unstack().reset_index()
        stacked.columns = ['Ticker', 'Date', feature_name]
        return stacked.set_index(['Date', 'Ticker'])
        
    df_master = stack_feature(weekly_etfs, 'Close')
    df_master = df_master.join(stack_feature(mom_1w, 'Mom_1W'))
    df_master = df_master.join(stack_feature(mom_13w, 'Mom_13W'))
    df_master = df_master.join(stack_feature(vol_13w, 'Vol_13W'))
    df_master = df_master.join(stack_feature(beta_26w, 'Beta_SPY_26W'))
    df_master = df_master.join(stack_feature(corr_tnx_26w, 'Corr_TNX_26W'))
    
    # 5. Targets (4-Week Forward Alpha and Returns)
    df_master['Log_Close'] = np.log(df_master['Close'])
    
    # 4-Week Forward simple returns for strategy tracking/returns
    df_master['Fwd_Ret_4W'] = df_master.groupby('Ticker')['Close'].shift(-4) / df_master['Close'] - 1
    
    # Raw 4-Week forward log returns for demeaning
    df_master['Raw_Fwd_4W'] = df_master.groupby('Ticker')['Log_Close'].shift(-4) - df_master['Log_Close']
    
    # Cross-Sectional demeaning to get raw alpha target (Zero future leakage across dates)
    df_master['Target_Alpha_4W'] = df_master['Raw_Fwd_4W'] - df_master.groupby('Date')['Raw_Fwd_4W'].transform('mean')
    
    # 6. Cross-Sectional Z-Scoring (Only on active sectors on each specific date)
    features_to_scale = ['Mom_1W', 'Mom_13W', 'Vol_13W', 'Beta_SPY_26W', 'Corr_TNX_26W']
    
    def cross_sectional_zscore(group):
        for col in features_to_scale:
            mean = group[col].mean()
            std = group[col].std() if group[col].std() != 0 else 1.0
            group[f'{col}_Z'] = (group[col] - mean) / std
        return group
        
    df_master = df_master.groupby(level='Date', group_keys=False).apply(cross_sectional_zscore)
    
    # 7. Clean up and flag inference rows
    df_master['Is_Training_Row'] = df_master['Target_Alpha_4W'].notna()
    
    # Remove early NaN rows from rolling indicators
    df_master = df_master.dropna(subset=['Corr_TNX_26W_Z'])
    
    return df_master
