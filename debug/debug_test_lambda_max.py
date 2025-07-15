from sklearn.datasets import make_regression
from skglm.experimental.quantile_huber import SmoothQuantileRegressor
from objective import Objective, pin_ball_loss
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

n_samples, n_features = 1000, 100
X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       n_informative=3, noise=0.5, random_state=0)

quantile = 0.5
fit_intercept = True
obj = Objective(reg=1.0, quantile=quantile, fit_intercept=fit_intercept)
obj.set_data(X, y)
lambda_max = obj._get_lambda_max(X, y)
print(f"lambda_max: {lambda_max:.4f}")

#  Smooth solver test
for factor in [1.0, 0.99]:
    lmbd = factor * lambda_max
    model = SmoothQuantileRegressor(quantile=quantile, alpha=lmbd,
                                    fit_intercept=fit_intercept)
    model.fit(X, y)
    beta = model.coef_
    n_nonzero = np.count_nonzero(beta)
    print(f"\nlambda = {factor} * lambda_max ({lmbd:.4f}):")
    print(f"  Nonzero coefficients: {n_nonzero}")
    print(f"  Coefficients: {beta}")
    if factor == 1.0:
        assert n_nonzero == 0, (
            "At lambda = lambda_max, all coefficients should be zero!")
    else:
        assert n_nonzero >= 1, (f"At lambda = {factor}*lambda_max, "
                                "at least one coefficient should be nonzero!")
print("Test passed: At lambda = lambda_max, all coefficients are zero! "
      "At lambda = 0.99*lambda_max, at least one coefficient is nonzero!")


# --- 2. Perturbation-based test ---
# TODO: need to verify and study more precisely how pertubation tests work
beta_0 = np.quantile(y, quantile) if fit_intercept else 0.0
beta_zero = np.zeros(n_features)
y_pred_zero = X @ beta_zero + beta_0
obj_val_zero = pin_ball_loss(y, y_pred_zero, quantile)
obj_val_zero += lambda_max * np.sum(np.abs(beta_zero))

# 1. At lambda_max: No direction should decrease the objective
eps = 1e-4
for j in range(n_features):
    beta_perturb = np.zeros(n_features)
    beta_perturb[j] = eps
    y_pred_perturb = X @ beta_perturb + beta_0
    obj_val_perturb = pin_ball_loss(y, y_pred_perturb, quantile)
    obj_val_perturb += lambda_max * np.sum(np.abs(beta_perturb))
    assert obj_val_perturb >= obj_val_zero - 1e-8, (
        f"At λ_max: Objective decreased for coordinate {j}!")
print("Perturbation test 1 passed: No direction decreases the objective at λ_max.")

# 2. At 0.99 * lambda_max: At least one direction should decrease the objective
lmbd_sub = 0.99 * lambda_max
obj_val_zero_sub = pin_ball_loss(y, y_pred_zero, quantile)
obj_val_zero_sub += lmbd_sub * np.sum(np.abs(beta_zero))
found_decreasing = False
for j in range(n_features):
    beta_perturb = np.zeros(n_features)
    beta_perturb[j] = eps
    y_pred_perturb = X @ beta_perturb + beta_0
    obj_val_perturb = pin_ball_loss(y, y_pred_perturb, quantile)
    obj_val_perturb += lmbd_sub * np.sum(np.abs(beta_perturb))
    diff = obj_val_perturb - obj_val_zero_sub
    if diff < -1e-8:
        print(f"At λ = 0.99*λ_max: Objective decreases for coordinate {j} "
              f"(Δobj = {diff})")
        found_decreasing = True
        break
assert found_decreasing, (
    "At 0.99*λ_max: No direction decreases the objective! λ_max may be too large.")
print("Perturbation test 2 passed: At least one direction decreases the objective "
      "at 0.99*λ_max.")

# #TODO: add KKT check
