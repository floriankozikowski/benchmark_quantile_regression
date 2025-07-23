from pathlib import Path
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.helpers.r_lang import import_func_from_r_file

# Setup R interface
R_FILE = str(Path(__file__).with_suffix('.R'))
import_func_from_r_file(R_FILE)
numpy2ri.activate()


class Solver(BaseSolver):
    name = "R-quantreg-lasso"
    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-quantreg']
    sampling_strategy = 'run_once'
    support_sparse = False

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X = X
        self.y = y
        self.lmbd = lmbd
        self.quantile = quantile
        self.fit_intercept = fit_intercept
        self.quantreg_lasso = robjects.r['quantreg_lasso']

    # no warm up needed, as solver only runs once

    def run(self, n_iter):
        # Convert to R objects
        X_r = robjects.r.matrix(self.X, nrow=self.X.shape[0], ncol=self.X.shape[1])
        y_r = robjects.FloatVector(self.y)

        # Run quantreg_lasso
        coefs = self.quantreg_lasso(
            X=X_r, y=y_r, tau=self.quantile, lmbd=self.lmbd,
            intercept=self.fit_intercept
        )

        # Convert back to numpy
        self.params = np.array(coefs)

    def get_result(self):
        return dict(params=self.params)
