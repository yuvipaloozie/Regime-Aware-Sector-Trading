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
            test_preds = hmm_model.predict(X_test_scaled)
        except Exception as e:
            # Fallback to simple state assignments if convergence fails
            print(f"HMM fit failure at {dates_test[0].date()}: {e}. Carrying forward state 1.")
            test_preds = np.ones(len(X_test_raw), dtype=int)
            
        out_of_sample_predictions.extend(test_preds)
        prediction_dates.extend(dates_test)
        
    # Build results DataFrame
    df_export = pd.DataFrame(index=prediction_dates)
    df_export.index.name = 'Date'
    df_export['Raw_OOS_Regime'] = out_of_sample_predictions
    
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
    conditions = [
        df_export['Smoothed_Regime'] == 0,
        df_export['Smoothed_Regime'] == 1,
        df_export['Smoothed_Regime'] == 2,
        df_export['Smoothed_Regime'] == 3
    ]
    choices = [1.0, 0.8, 0.5, 0.0]
    df_export['Risk_Scalar'] = np.select(conditions, choices, default=1.0)
    
    return df_export
