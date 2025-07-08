from benchopt import BaseObjective
import numpy as np
from numpy.linalg import norm


def pin_ball_loss(y_true, y_pred, quantile=0.5):
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = quantile * sign * diff - (1 - quantile) * (1 - sign) * diff
    return np.mean(loss)


class Objective(BaseObjective):
    min_benchopt_version = "1.6"
    name = "L1-regularized Quantile Regression"

    parameters = {
        'reg': [.5, .1, .05],
        'quantile': [0.2, 0.5, 0.8],
        'fit_intercept': [True, False]
    }

    def __init__(self, reg=0., quantile=0.5, fit_intercept=True):
        self.reg = reg
        self.quantile = quantile
        self.fit_intercept = fit_intercept

    def set_data(self, X, y):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max(X, y)

    def evaluate_result(self, params):
        n_features = self.X.shape[1]
        beta = params[:n_features]
        intercept = params[-1] if self.fit_intercept else 0.

        y_pred = self.X.dot(beta) + intercept
        l1 = np.sum(np.abs(beta))
        pobj = pin_ball_loss(self.y, y_pred, self.quantile) + self.lmbd * l1
        return dict(value=pobj)

    def _get_lambda_max(self, X, y):
        # optimality condition for w = 0.
        #   for all g in subdiff pinball(y), g must be in subdiff ||.||_1(0)
        # hint: consider max(x, 0) = (x + |x|) / 2 to compute subdiff pinball

        if self.fit_intercept:
            # Optimal intercept when beta=0 is the tau-quantile of y
            beta_0_star = np.quantile(y, self.quantile)
            residuals = y - beta_0_star
            # Compute subdifferential at these residuals
            subdiff_zero = np.sign(residuals)/2 + (self.quantile - 1/2)
        else:
            # No intercept case
            subdiff_zero = np.sign(y)/2 + (self.quantile - 1/2)
        lmbd_max = norm(X.T @ subdiff_zero, ord=np.inf) / len(y)

        return lmbd_max

    def get_one_result(self):
        n_features = self.X.shape[1]
        if self.fit_intercept:
            n_features += 1
        return dict(params=np.zeros(n_features))

    def get_objective(self):
        return dict(
            X=self.X, y=self.y, lmbd=self.lmbd, quantile=self.quantile,
            fit_intercept=self.fit_intercept)
