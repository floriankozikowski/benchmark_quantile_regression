from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import statsmodels.api as sm


class Solver(BaseSolver):
    name = "statsmodels-qr"
    install_cmd = 'conda'
    requirements = ['statsmodels', 'pandas']
    sampling_strategy = 'iteration'

    def skip(self, X, y, lmbd, quantile, fit_intercept):
        # This solver does not support regularization
        if lmbd > 0:
            return True, "statsmodels.QuantReg does not support regularization"
        # This is an unpenalized solver, so it may fail for n_features > n_samples
        if X.shape[1] > X.shape[0]:
            return True, "Unpenalized solver does not support n_features > n_samples"
        return False, None

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y = X, y
        self.quantile = quantile
        self.fit_intercept = fit_intercept

        if self.fit_intercept:
            self.X_fit = sm.add_constant(self.X)
        else:
            self.X_fit = self.X

        self.model = sm.QuantReg(self.y, self.X_fit)

    def run(self, n_iter):
        if n_iter == 0:
            p = self.X_fit.shape[1]
            self.beta = np.zeros(p)
            return

        # We fit the model with at most n_iter iterations.
        res = self.model.fit(q=self.quantile, max_iter=n_iter)
        self.beta = res.params

    def get_result(self):
        return dict(params=self.beta)
