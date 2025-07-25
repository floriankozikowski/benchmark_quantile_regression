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
    sampling_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        # ensure attributes exist even if run() is skipped (cache hit)
        self.coef_ = None
        self.intercept_ = 0.0

    def run(self, tol):
        tol = max(tol, 1e-4)
        est = QuantileRegressor(
            quantile=self.quantile,
            alpha=self.lmbd,
            fit_intercept=self.fit_intercept,
            solver="highs",
            solver_options={
                "time_limit": 300,  # 5 minutes
                "dual_feasibility_tolerance": tol
            }
        )
        est.fit(self.X, self.y)
        if est.coef_ is None:        # HiGHS aborted
            raise RuntimeError(
                "HiGHS time-limit reached before a feasible solution was found.")
        self.coef_ = est.coef_
        self.intercept_ = est.intercept_ if self.fit_intercept else 0.0

    def get_result(self):
        # fallback when run() never executed in this Python session
        if getattr(self, "coef_", None) is None:
            self.coef_ = np.zeros(self.X.shape[1])
            self.intercept_ = (
                np.quantile(self.y, self.quantile) if self.fit_intercept else 0.0
            )

        if self.fit_intercept:
            params = np.concatenate((self.coef_, [self.intercept_]))
        else:
            params = self.coef_
        return dict(params=params)
