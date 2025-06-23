from benchopt import BaseSolver, safe_import_context
import numpy as np

with safe_import_context() as import_ctx:
    from sklearn.linear_model import QuantileRegressor


class Solver(BaseSolver):
    """Quantile regression solver using scikit-learn's QuantileRegressor."""
    name = 'sklearn-QuantileRegressor'

    install_cmd = 'conda'
    requirements = ['numpy', 'scikit-learn']
    parameters = {
        # To benchmark different solvers, uncomment the following line:
        # 'solver': ['highs-ds', 'highs-ipm', 'highs',
        #            'interior-point', 'revised simplex'],
    }
    stop_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def run(self, tol):
        est = QuantileRegressor(
            quantile=self.quantile,
            alpha=self.lmbd,
            fit_intercept=self.fit_intercept,
            solver='highs',  # change to solver=self.solver if parameter is enabled
        )
        est.fit(self.X, self.y)
        self.coef_ = est.coef_
        self.intercept_ = est.intercept_ if self.fit_intercept else 0.0

    def get_result(self):
        params = np.concatenate((self.coef_, [self.intercept_]))
        return dict(params=params)
