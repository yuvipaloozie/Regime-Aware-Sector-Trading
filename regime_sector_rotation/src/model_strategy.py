import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm.auto import tqdm


BASE_FEATURES = ["Mom_1W", "Mom_13W", "Vol_13W", "Beta_SPY_26W", "Corr_TNX_26W"]


def iter_purged_walk_forward_splits(all_dates, min_train, purge, step):
    """Yield train/test dates with a strict label-horizon embargo."""
    all_dates = pd.Index(all_dates).sort_values().unique()
    for test_start in range(min_train + purge, len(all_dates), step):
        # Excluding the boundary is essential: its forward label ends at or after
        # the first test observation and is not knowable when that signal is made.
        train_dates = all_dates[: test_start - purge]
        test_dates = all_dates[test_start : test_start + step]
        if len(train_dates) >= min_train and len(test_dates):
            yield train_dates, test_dates


def run_walk_forward_xgboost(df_master, config):
    """Fit purged expanding rankers and include unlabeled live inference rows."""
    xgb_cfg = config["xgboost"]
    probability_features = sorted(c for c in df_master.columns if c.startswith("Regime_Prob_"))
    features = [f"{column}_Z" for column in BASE_FEATURES] + probability_features
    if not probability_features:
        features.append("Smoothed_Regime")

    data = df_master.sort_index().copy()
    all_dates = data.index.get_level_values("Date").unique().sort_values()
    params = {
        "objective": "rank:pairwise",
        "learning_rate": xgb_cfg["learning_rate"],
        "max_depth": xgb_cfg["max_depth"],
        "n_estimators": xgb_cfg["n_estimators"],
        "subsample": xgb_cfg["subsample"],
        "colsample_bytree": xgb_cfg["colsample_bytree"],
        "random_state": xgb_cfg["random_state"],
        "n_jobs": -1,
    }
    predictions = []
    importances = []
    splits = iter_purged_walk_forward_splits(
        all_dates, xgb_cfg["min_train_weeks"], xgb_cfg["purge_weeks"], xgb_cfg["step_weeks"]
    )
    for train_dates, test_dates in tqdm(list(splits), desc="XGBoost folds"):
        train = data[data.index.get_level_values("Date").isin(train_dates)]
        train = train[train["Is_Training_Row"]].dropna(subset=features + ["Target_Alpha_4W"])
        test = data[data.index.get_level_values("Date").isin(test_dates)].dropna(subset=features)
        if train.empty or test.empty:
            continue
        groups = train.groupby(level="Date", sort=True).size().to_numpy()
        model = xgb.XGBRanker(**params)
        model.fit(train[features], train["Target_Alpha_4W"], group=groups, verbose=False)
        frame = test[["Raw_Fwd_4W", "Target_Alpha_4W", "Smoothed_Regime", "Risk_Scalar"]].copy()
        frame["ML_Rank_Score"] = model.predict(test[features])
        predictions.append(frame)
        importances.append(model.feature_importances_)

    if not predictions:
        raise ValueError("No predictions generated. Check date length and walk-forward configuration.")
    results = pd.concat(predictions).sort_index()
    average = np.mean(importances, axis=0)
    return results, dict(zip(features, average))


def evaluate_rank_predictions(oos_results, top_n=2):
    """Evaluate only observations whose forward outcomes are now known."""
    labeled = oos_results.dropna(subset=["Target_Alpha_4W", "ML_Rank_Score"])
    rank_ics = []
    hit_rates = []
    for _, group in labeled.groupby(level="Date"):
        if len(group) < 2:
            continue
        rank_ics.append(group["ML_Rank_Score"].corr(group["Target_Alpha_4W"], method="spearman"))
        predicted = set(group.nlargest(top_n, "ML_Rank_Score").index.get_level_values("Ticker"))
        realized = set(group.nlargest(top_n, "Target_Alpha_4W").index.get_level_values("Ticker"))
        hit_rates.append(len(predicted & realized) / top_n)
    clean_ic = [value for value in rank_ics if pd.notna(value)]
    return {
        "Mean Rank IC": float(np.mean(clean_ic)) if clean_ic else np.nan,
        "Top-N Hit Rate": float(np.mean(hit_rates)) if hit_rates else np.nan,
        "Evaluated Dates": len(hit_rates),
    }


def _defensive_choice(date, df_weekly, assets, lookback, minimum_return, cash_symbol):
    history = df_weekly.loc[:date, assets].tail(lookback + 1)
    if len(history) < lookback + 1:
        return cash_symbol
    returns = history.pct_change().dropna()
    total = history.iloc[-1] / history.iloc[0] - 1
    volatility = returns.std().replace(0, np.nan)
    scores = (total / volatility).replace([np.inf, -np.inf], np.nan).dropna()
    if scores.empty:
        return cash_symbol
    winner = scores.idxmax()
    return winner if total[winner] > minimum_return else cash_symbol


def calculate_portfolio_weights(oos_results, df_master, config, df_weekly=None):
    """Select sectors with hysteresis and a causal defensive sleeve."""
    portfolio = config["portfolio"]
    top_n, buffer = portfolio["top_n"], portfolio["buffer"]
    dates = oos_results.index.get_level_values("Date").unique().sort_values()
    tickers = list(config["sectors"])
    defensive = list(config.get("defensive_assets", [config["benchmarks"]["safe_haven"]]))
    cash_symbol = portfolio.get("cash_symbol", "CASH")
    columns = tickers + defensive + [cash_symbol]
    weights = pd.DataFrame(0.0, index=dates[:: portfolio.get("rebalance_every_weeks", 4)], columns=columns)
    current = []

    for date in weights.index:
        day = oos_results.xs(date, level="Date").copy()
        day["Rank"] = day["ML_Rank_Score"].rank(ascending=False, method="first")
        risk_scalar = float(day["Risk_Scalar"].iloc[0])
        retained = [ticker for ticker in current if ticker in day.index and day.loc[ticker, "Rank"] <= top_n + buffer]
        buys = day[~day.index.isin(retained)].sort_values("Rank").head(top_n - len(retained)).index.tolist()
        current = retained + buys
        if current:
            weights.loc[date, current] = risk_scalar / len(current)

        defensive_weight = 1.0 - risk_scalar
        if defensive_weight > 0:
            selected = cash_symbol
            if df_weekly is not None and set(defensive).issubset(df_weekly.columns):
                selected = _defensive_choice(
                    date, df_weekly, defensive, portfolio.get("defensive_lookback_weeks", 13),
                    portfolio.get("defensive_min_return", 0.0), cash_symbol,
                )
            weights.loc[date, selected] = defensive_weight
    return weights


def build_execution_schedule(weight_matrix, daily_index, lag_trading_days=1):
    """Map signal dates to later executable close dates."""
    daily_index = pd.DatetimeIndex(daily_index).sort_values().unique()
    rows, execution_dates, signal_dates = [], [], []
    for signal_date, row in weight_matrix.iterrows():
        later = daily_index[daily_index > pd.Timestamp(signal_date)]
        if len(later) < lag_trading_days:
            continue
        execution_dates.append(later[lag_trading_days - 1])
        signal_dates.append(pd.Timestamp(signal_date))
        rows.append(row.to_numpy())
    schedule = pd.DataFrame(rows, index=execution_dates, columns=weight_matrix.columns)
    schedule.index.name = "ExecutionDate"
    return schedule, pd.Series(signal_dates, index=execution_dates, name="SignalDate")


def _performance_metrics(returns, annual_risk_free_rate, periods=252):
    returns = returns.dropna()
    wealth = (1 + returns).cumprod()
    rf_period = (1 + annual_risk_free_rate) ** (1 / periods) - 1
    excess = returns - rf_period
    years = len(returns) / periods
    return {
        "Ann. Sharpe": float(np.sqrt(periods) * excess.mean() / excess.std()) if excess.std() > 0 else 0.0,
        "Strategy Total Return (%)": float((wealth.iloc[-1] - 1) * 100) if len(wealth) else 0.0,
        "Strategy CAGR (%)": float((wealth.iloc[-1] ** (1 / years) - 1) * 100) if years > 0 else 0.0,
        "Max Drawdown (%)": float(((wealth / wealth.cummax()) - 1).min() * 100) if len(wealth) else 0.0,
    }


def _momentum_baseline(prices, sectors, lookback=63, top_n=2):
    scores = prices[sectors].pct_change(lookback)
    weights = scores.rank(axis=1, ascending=False, method="first").le(top_n).astype(float) / top_n
    return (weights.shift(1) * prices[sectors].pct_change()).sum(axis=1)


def backtest_strategy(weight_matrix, df_master, df_weekly, config, df_prices_daily=None):
    """Daily mark-to-market backtest with delayed execution and explicit costs."""
    if df_prices_daily is None:
        # Compatibility fallback for tests; production passes adjusted daily closes.
        df_prices_daily = df_weekly
    prices = df_prices_daily.sort_index().copy()
    portfolio = config["portfolio"]
    cash_symbol = portfolio.get("cash_symbol", "CASH")
    annual_rf = portfolio.get("annual_risk_free_rate", 0.0)
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    if cash_symbol not in prices:
        prices[cash_symbol] = (1 + daily_rf) ** np.arange(len(prices))
    prices = prices.reindex(columns=weight_matrix.columns.union(pd.Index([config["benchmarks"]["equity"]]))).ffill()

    schedule, signal_dates = build_execution_schedule(
        weight_matrix, prices.index, portfolio.get("execution_lag_trading_days", 1)
    )
    if schedule.empty:
        raise ValueError("No signal has a later executable market date.")
    daily_weights = schedule.reindex(prices.index).ffill().fillna(0.0)
    asset_returns = prices[daily_weights.columns].pct_change(fill_method=None).fillna(0.0)
    gross_returns = (daily_weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
    turnover = daily_weights.diff().abs().sum(axis=1).fillna(0.0)
    cost_rate = portfolio.get("transaction_cost_bps", 0.0) / 10000.0
    net_returns = gross_returns - turnover * cost_rate

    sectors = config["sectors"]
    sector_returns = prices[sectors].pct_change(fill_method=None)
    equal_weight = sector_returns.mean(axis=1, skipna=True).fillna(0.0)
    spy_returns = prices[config["benchmarks"]["equity"]].pct_change(fill_method=None).fillna(0.0)
    momentum_returns = _momentum_baseline(prices, sectors).fillna(0.0)

    # All strategies must share the same investable evaluation interval.
    evaluation_index = prices.index[prices.index >= schedule.index.min()]
    net_returns = net_returns.reindex(evaluation_index)
    gross_returns = gross_returns.reindex(evaluation_index)
    turnover = turnover.reindex(evaluation_index)
    equal_weight = equal_weight.reindex(evaluation_index)
    spy_returns = spy_returns.reindex(evaluation_index)
    momentum_returns = momentum_returns.reindex(evaluation_index)
    daily_weights = daily_weights.reindex(evaluation_index)
    metrics = _performance_metrics(net_returns, annual_rf)
    metrics["Benchmark Total Return (%)"] = float(((1 + equal_weight).prod() - 1) * 100)
    metrics["SPY Total Return (%)"] = float(((1 + spy_returns).prod() - 1) * 100)
    metrics["Momentum Total Return (%)"] = float(((1 + momentum_returns).prod() - 1) * 100)
    metrics["Avg Annual Turnover (%)"] = float(turnover.mean() * 252 * 100)

    cost_sensitivity = {}
    for bps in (0, 5, 10, 20):
        scenario = gross_returns - turnover * bps / 10000.0
        cost_sensitivity[str(bps)] = _performance_metrics(scenario, annual_rf)

    return {
        "net_returns": net_returns,
        "benchmark_returns": equal_weight,
        "spy_returns": spy_returns,
        "momentum_returns": momentum_returns,
        "cum_strategy": (1 + net_returns).cumprod(),
        "cum_benchmark": (1 + equal_weight).cumprod(),
        "cum_spy": (1 + spy_returns).cumprod(),
        "cum_momentum": (1 + momentum_returns).cumprod(),
        "turnover": turnover,
        "daily_weights": daily_weights,
        "execution_schedule": schedule,
        "execution_signal_dates": signal_dates,
        "metrics": metrics,
        "cost_sensitivity": cost_sensitivity,
    }
