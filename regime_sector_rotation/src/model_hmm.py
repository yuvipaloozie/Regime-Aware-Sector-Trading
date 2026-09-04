import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings
import logging

# Mute noisy training logs from hmmlearn
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class AnchoredGaussianHMM(BaseEstimator, ClassifierMixin):
    """
    Custom Scikit-Learn wrapper around hmmlearn's GaussianHMM.
    Post-fit, it permutes transition, start, mean, and covariance parameters 
    so that hidden states are sorted monotonically by the VIX level mean (vix_idx).
    This guarantees that:
      State 0 = Lowest VIX (lowest volatility)
      State N-1 = Highest VIX (highest volatility / crisis)
    """
    def __init__(self, n_components=4, covariance_type="diag", n_iter=1000, random_state=42, vix_idx=1):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.vix_idx = vix_idx
        self.model_ = None

    def fit(self, X, y=None):
        # Create and fit the raw GaussianHMM
        self.model_ = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.model_.fit(X)
        
        # Extract VIX means for each latent state
        vix_means = self.model_.means_[:, self.vix_idx]
        
        # Create a permutation array P that sorts VIX means in ascending order
        P = np.argsort(vix_means)
        
        # Re-order HMM parameters to anchor hidden states
        self.model_.startprob_ = self.model_.startprob_[P]
        self.model_.transmat_ = self.model_.transmat_[P][:, P]
        self.model_.means_ = self.model_.means_[P]
        self.model_._covars_ = self.model_._covars_[P]
            
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_.predict_proba(X)

    def score(self, X, y=None):
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        return self.model_.score(X)


def _normalize_probability(values):
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return values / total


def causal_filter_probabilities(fitted_model, X_history, X_new):
    """Filter states sequentially so no future observation changes an earlier state.

    ``hmmlearn.predict`` performs sequence decoding. Calling it on a multi-row test
    block lets later rows affect earlier decoded states. This function applies the
    forward filter and carries the posterior from history into each new row.
    """
    model = fitted_model.model_ if isinstance(fitted_model, AnchoredGaussianHMM) else fitted_model
    if model is None:
        raise ValueError("Model is not fitted yet.")

    history_ll = model._compute_log_likelihood(np.asarray(X_history))
    alpha = _normalize_probability(model.startprob_ * np.exp(history_ll[0] - history_ll[0].max()))
    for log_likelihood in history_ll[1:]:
        emission = np.exp(log_likelihood - log_likelihood.max())
        alpha = _normalize_probability((alpha @ model.transmat_) * emission)

    probabilities = []
    for row in np.asarray(X_new):
        log_likelihood = model._compute_log_likelihood(row.reshape(1, -1))[0]
        emission = np.exp(log_likelihood - log_likelihood.max())
        alpha = _normalize_probability((alpha @ model.transmat_) * emission)
        probabilities.append(alpha.copy())
    return np.asarray(probabilities)


def run_walk_forward_hmm(df_features, config):
    """
    Executes a walk-forward out-of-sample HMM classification.
    Zero-leakage: fits StandardScaler on training data, scales test data.
    """
    hmm_cfg = config['hmm']
    n_states = hmm_cfg['n_components']
    window_size = hmm_cfg['window_size']
    step_size = hmm_cfg['step_size']
    vix_idx = hmm_cfg['vix_idx']
    cov_type = hmm_cfg['covariance_type']
    n_iter = hmm_cfg['n_iter']
    random_state = hmm_cfg['random_state']
    
    features_to_use = ['SPY_Log_Ret_W', 'VIX_Level', 'Yield_Curve_Change_W', 'Financial_Stress', 'Inflation_MoM']
    
    feature_array = df_features[features_to_use].values
    dates_array = df_features.index
    
    out_of_sample_predictions = []
    out_of_sample_probabilities = []
    prediction_dates = []
    
    print(f"Running rolling walk-forward HMM on {len(df_features)} weeks...")
    
    for t in range(window_size, len(df_features), step_size):
        # 1. Slice training and test blocks
        X_train_raw = feature_array[t - window_size : t]
        
        end_idx = min(t + step_size, len(df_features))
        X_test_raw = feature_array[t : end_idx]
        dates_test = dates_array[t : end_idx]
        
        # 2. Fit scaler strictly on training block (Zero leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # 3. Fit the custom Anchored HMM
        hmm_model = AnchoredGaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=n_iter,
            random_state=random_state,
            vix_idx=vix_idx
        )
        
        try:
            hmm_model.fit(X_train_scaled)
            test_probabilities = causal_filter_probabilities(hmm_model, X_train_scaled, X_test_scaled)
            test_preds = test_probabilities.argmax(axis=1)
        except Exception as e:
            # Fallback to simple state assignments if convergence fails
            print(f"HMM fit failure at {dates_test[0].date()}: {e}. Carrying forward state 1.")
            test_preds = np.ones(len(X_test_raw), dtype=int)
            test_probabilities = np.zeros((len(X_test_raw), n_states), dtype=float)
            test_probabilities[:, min(1, n_states - 1)] = 1.0
            
        out_of_sample_predictions.extend(test_preds)
        out_of_sample_probabilities.extend(test_probabilities)
        prediction_dates.extend(dates_test)
        
    # Build results DataFrame
    df_export = pd.DataFrame(index=prediction_dates)
    df_export.index.name = 'Date'
    df_export['Raw_OOS_Regime'] = out_of_sample_predictions
    for state in range(n_states):
        df_export[f'Regime_Prob_{state}'] = np.asarray(out_of_sample_probabilities)[:, state]
    
    # 4. SIGNAL SMOOTHING (Hysteresis Filter)
    # Mode over a 3-week rolling window prevents rapid whiplash rebalancing
    df_export['Smoothed_Regime'] = (
        df_export['Raw_OOS_Regime']
        .rolling(window=3)
        .apply(lambda x: pd.Series(x).mode()[0])
        .fillna(df_export['Raw_OOS_Regime'])
        .astype(int)
    )
    
    # 5. RISK SCALING OVERLAY
    # State 0 (lowest vol): 1.0 risk exposure
    # State 1 (warning):    0.8 risk exposure
    # State 2 (crisis):     0.5 risk exposure
    # State 3 (severe):     0.0 risk exposure (complete cash/TLT defense)
    choices = np.asarray([1.0, 0.8, 0.5, 0.0][:n_states], dtype=float)
    probability_columns = [f'Regime_Prob_{state}' for state in range(n_states)]
    df_export['Risk_Scalar'] = df_export[probability_columns].to_numpy() @ choices
    
    return df_export
