from scipy import optimize
import pandas as pd
import numpy as np
from pypfopt.risk_models import risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier


class MarkowitzOptimizer:
    def __init__(self, data, return_cols, window_cov=10, max_quadratic=False):
        self.data_ = data[["Date"]+return_cols].dropna().sort_values("Date").copy()
        self.window_cov = window_cov
        self.return_cols = return_cols
        self.n_features = len(return_cols)
        self.max_quadratic = max_quadratic
        
    def execute_markowitz(self):
        cons = {'type':'eq','fun':self._check_sum}
        bounds = tuple((0,1) for _ in range(self.n_features))  # weights bounds
        init_guess = [1/self.n_features for _ in range(self.n_features)] ## initial guess of weiths
        feature_result_names = [f"optimal_{x}" for x in self.return_cols]
        self.data_[feature_result_names] = np.nan

        for i in range(self.window_cov,len(self.data_)):
            self.returns = self.data_.iloc[i][self.return_cols].values
            self.cov = risk_matrix(
                prices=self.data_.iloc[i-self.window_cov:i,:][self.return_cols],
                returns_data=True,
                method="sample_cov"
            )
            if not self.max_quadratic:
                opt_results = optimize.minimize(self._neg_sr, init_guess, constraints=cons, bounds=bounds, method='SLSQP')
                optimal_weights = opt_results.x
            else:
                # max quadratic will return weights from -1 to 1 (short and long)
                ef = EfficientFrontier(self.returns, self.cov)
                weights = ef.max_quadratic_utility(risk_aversion=1,market_neutral=True)
                weights = dict(weights)
                optimal_weights = weights.values()
            self.data_.iloc[i,-self.n_features:] = [float(x) for x in optimal_weights]
            
        self.data_[feature_result_names] = self.data_[feature_result_names].astype(float).round(6) 
        self.data_["sum_weights"] = self.data_[feature_result_names].sum(axis=1)
        self.feature_result_names = feature_result_names
        return self.data_
    
    def _get_ret_vol_sr(self, weights):
        weights = np.array(weights)
        ret = self.returns.dot(weights)
        vol = np.sqrt(weights.T.dot(self.cov.dot(weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])

    def _neg_sr(self, weights):
        return self._get_ret_vol_sr(weights)[-1] * -1
        
    @staticmethod
    def _check_sum(weights):
        return np.sum(weights) - 1