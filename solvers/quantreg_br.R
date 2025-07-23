#' Run quantreg quantile regression with Barrodale-Roberts method
#' @param X
#' @param y
#' @param tau
#' @param lmbd
#' @param intercept
#' @return
quantreg_br <- function(X, y, tau, lmbd, intercept = TRUE) {

  library(quantreg)

  data <- data.frame(y = y, X = X)

  if (intercept) {
    # quantreg with formula y ~ . fits an intercept.
    # The intercept is the first coefficient.
    fit <- rq(y ~ ., data = data, tau = tau, method = "br")
    beta_all <- coef(fit)
    # The output of benchopt is expected to have the intercept at the end.
    # Reorder from [intercept, betas] to [betas, intercept]
    intercept_val <- beta_all[1]
    coeff_vals <- beta_all[-1]
    params <- c(coeff_vals, intercept_val)
  } else {
    # y ~ 0 + . fits no intercept.
    fit <- rq(y ~ 0 + ., data = data, tau = tau, method = "br")
    params <- coef(fit)
  }

  return(as.numeric(params))
}