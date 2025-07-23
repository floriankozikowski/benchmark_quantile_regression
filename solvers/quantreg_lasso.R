#' Run L1-penalized quantreg quantile regression
#' @param X Numeric matrix of predictors
#' @param y Numeric vector of response
#' @param tau Quantile level (between 0 and 1)
#' @param lmbd Regularization parameter
#' @param intercept Logical, whether to fit intercept
#' @return Coefficient vector
quantreg_lasso <- function(X, y, tau, lmbd, intercept = TRUE) {
  # Load the quantreg library
  library(quantreg)

  if (!intercept) {
    # If no intercept is requested, we fit on centered data,
    # which is not standard for this solver but necessary
    # to match the no-intercept convention.
    y_mean <- mean(y)
    x_means <- colMeans(X)
    X <- scale(X, center = x_means, scale = FALSE)
    y <- y - y_mean
  }

  # `rq.fit.lasso` performs L1-penalized quantile regression.
  # The lambda parameter in `rq.fit.lasso` corresponds to `lmbd`.
  fit <- rq.fit.lasso(X, y, tau = tau, lambda = lmbd)

  # The coefficients are returned in `fit$coef`.
  # The first element is the intercept.
  beta_all <- fit$coef

  if (intercept) {
    # Reorder from [intercept, betas] to [betas, intercept]
    intercept_val <- beta_all[1]
    coeff_vals <- beta_all[-1]
    params <- c(coeff_vals, intercept_val)
  } else {
    # If no intercept was fitted, we return the coefficients as is,
    # and add the intercept computed from the means.
    params <- beta_all
  }

  return(as.numeric(params))
}