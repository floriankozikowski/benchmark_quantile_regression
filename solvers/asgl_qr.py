from benchopt import BaseSolver, safe_import_context
import numpy as np
import warnings

with safe_import_context() as import_ctx:
    from asgl import Regressor


class Solver(BaseSolver):
    """Quantile regression solver using asgl."""
    name = 'asgl-QuantileRegressor'

    install_cmd = 'pip'
    requirements = ["numpy", "scikit-learn", "asgl"]
    parameters = {
        "penalization": ["lasso"],
    }
    sampling_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def run(self, tol):
        # asgl uses lambda1 for L1 penalty
        reg = Regressor(
            model='qr',
            penalization=self.penalization,
            lambda1=self.lmbd,
            quantile=self.quantile,
            fit_intercept=self.fit_intercept,
        )
        warnings.filterwarnings('ignore')
        reg.fit(self.X, self.y)
        self.coef_ = reg.coef_
        if self.fit_intercept:
            self.intercept_ = reg.intercept_

    def get_result(self):
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
