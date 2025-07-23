# solvers/solver_skglm_quantile_huber.py

from benchopt import BaseSolver, safe_import_context
import numpy as np
import warnings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
with safe_import_context() as import_ctx:
    from huberplayground import SmoothQuantileRegressorPlayground


class Solver(BaseSolver):
    """Smooth quantile regression solver using skglm."""
    name = 'skglm-Playground'

    install_cmd = 'conda'
    requirements = ["numpy", "scikit-learn", "numba", "skglm"]
    parameters = {}
    sampling_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        # ensure attributes exist even if run() is skipped (cache hit)
        self.coef_ = None
        self.intercept_ = 0.0

    def warm_up(self):
        # Cache pre-compilation and other one-time setups that should
        # not be included in the benchmark timing.
        self.run(1)  # For sampling_strategy == 'tolerance' or 'iteration'

    def run(self, tol):
        # Main fit (timed by Benchopt)
        est = SmoothQuantileRegressorPlayground(
            quantile=self.quantile,
            alpha=self.lmbd,
            delta_init=1,
            delta_final=0.0001,
            max_iter=516,
            tol=max(tol, 1e-4),
            verbose=False,
            fit_intercept=self.fit_intercept,
        )
        warnings.filterwarnings('ignore')
        est.fit(self.X, self.y)

        self.coef_ = est.coef_
        if self.fit_intercept:
            self.intercept_ = est.intercept_

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
