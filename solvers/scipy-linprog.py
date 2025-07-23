import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import linprog
    from scipy.optimize import OptimizeWarning
    from scipy.linalg import LinAlgWarning
    from scipy import sparse


def quantile_regression(X, y, quantile, lmbd, tol, solver, fit_intercept=True):
    n_samples, n_features = X.shape
    sample_weights = np.ones(n_samples) / n_samples
    n_params = n_features
    if fit_intercept:
        n_params += 1

    # the linear programming formulation of quantile regression
    # follows https://stats.stackexchange.com/questions/384909/
    #
    # The objective is defined as 1/n * sum(pinball loss) + alpha * L1.
    # So we rescale the penalty term, which is equivalent.
    c = np.concatenate([
        np.ones(n_params * 2) * lmbd,
        sample_weights * quantile,
        sample_weights * (1 - quantile),
    ])

    if fit_intercept:
        # do not penalize the intercept
        c[0] = 0
        c[n_params] = 0

        A_eq = np.concatenate([
            np.ones((n_samples, 1)),
            X,
            -np.ones((n_samples, 1)),
            -X,
            np.eye(n_samples),
            -np.eye(n_samples),
        ], axis=1)
    else:
        A_eq = np.concatenate([
            X,
            -X,
            np.eye(n_samples),
            -np.eye(n_samples),
        ], axis=1)

    if sparse.issparse(X):
        A_eq = sparse.csc_matrix(A_eq)

    b_eq = y

    if 'highs' in solver:
        options = {'primal_feasibility_tolerance': tol}
    else:
        options = {'tol': tol}

    warnings.filterwarnings('ignore', category=OptimizeWarning)
    warnings.filterwarnings('ignore', category=LinAlgWarning)

    result = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        method=solver,
        options=options,
    )
    solution = result.x
    params = solution[:n_params] - solution[n_params:2 * n_params]

    if fit_intercept:
        coef_ = params[1:]
        intercept_ = params[0]
    else:
        coef_ = params
        intercept_ = 0.0

    return coef_, intercept_


class Solver(BaseSolver):
    """Simplex solver using scipy linprog function."""
    name = 'scipy-linprog'

    install_cmd = 'conda'
    requirements = [
        'numpy',
        'scipy'
    ]
    parameters = {
        'solver': [
            'simplex', 'interior-point', 'revised simplex',
            'highs', 'highs-ipm', 'highs-ds'
        ],
    }
    stop_strategy = 'tolerance'

    def set_objective(self, X, y, lmbd, quantile, fit_intercept):
        self.X, self.y, self.lmbd, self.quantile = X, y, lmbd, quantile
        self.fit_intercept = fit_intercept

    def run(self, tol):
        tol = 1e-3 * tol
        self.coef_, self.intercept_ = quantile_regression(
            self.X, self.y, self.quantile, self.lmbd, tol,
            self.solver, fit_intercept=self.fit_intercept
        )

    def get_result(self):
        params = np.concatenate((self.coef_, [self.intercept_]))
        return dict(params=params)
