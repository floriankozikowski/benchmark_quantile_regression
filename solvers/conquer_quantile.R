#' Run conquer quantile regression
#' @param X
#' @param y
#' @param tau
#' @param lmbd
#' @param intercept
#' @return
conquer_quantile <- function(X, y, tau, lmbd, intercept = TRUE) {

  library(conquer)

  # conquer.reg is assumed to always fit an intercept and return it as the
  # first coefficient.
  fit <- conquer.reg(
    X = X,
    Y = y,
    tau = tau,
    lambda = lmbd
  )

  # The coefficients are returned as a numeric vector.
  # The first element is the intercept, and the rest are the slope coefficients.
  beta_all <- as.numeric(fit$coeff)

  if (intercept) {
    # The output of benchopt is expected to have the intercept at the end.
    # Reorder from [intercept, betas] to [betas, intercept]
    intercept_val <- beta_all[1]
    coeff_vals <- beta_all[-1]
    params <- c(coeff_vals, intercept_val)
  } else {
    # If no intercept was requested, return only the slope coefficients.
    # The intercept fitted by conquer.reg is discarded.
    params <- beta_all[-1]
  }

  return(as.numeric(params))
}