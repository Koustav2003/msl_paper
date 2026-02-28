# Load libraries
if (!require(mclust)) install.packages("mclust", repos="http://cran.us.r-project.org")
library(mclust)

dir.create("experiments", showWarnings = FALSE)

set.seed(42)

# --- Folded Normal Experiment ---
generate_folded_normal <- function(mu, sigma, n) {
  x <- rnorm(n, mean = mu, sd = sigma)
  return(abs(x))
}

orbit_dist_fn <- function(mu1, sigma1, mu2, sigma2) {
  d1 <- sqrt((mu1 - mu2)^2 + (sigma1 - sigma2)^2)
  d2 <- sqrt((mu1 + mu2)^2 + (sigma1 - sigma2)^2)
  return(min(d1, d2))
}

moment_estimator_fn <- function(data) {
  m2 <- mean(data^2)
  m4 <- mean(data^4)
  val <- (3 * m2^2 - m4) / 2
  u_hat <- sqrt(max(0, val))
  v_hat <- m2 - u_hat
  mu_hat <- sqrt(max(0, u_hat))
  sigma_hat <- sqrt(max(0, v_hat))
  return(c(mu_hat, sigma_hat))
}

net_erm_fn <- function(data, m_grid, s_grid) {
  best_loss <- Inf
  best_params <- c(0, 1)

  for (mu in m_grid) {
    for (sigma in s_grid) {
      x <- (mu * data) / (sigma^2)
      log_cosh <- ifelse(abs(x) < 50, log(cosh(x)), abs(x) - log(2.0))

      loss <- mean(log(sigma * sqrt(2 * pi) / 2) + (data^2 + mu^2) / (2 * sigma^2) - log_cosh)

      if (loss < best_loss) {
        best_loss <- loss
        best_params <- c(mu, sigma)
      }
    }
  }
  return(best_params)
}

run_folded_normal_experiments <- function() {
  true_mu <- 1.0
  true_sigma <- 1.0

  sample_sizes <- c(100, 500, 1000, 2000, 5000, 10000)
  moment_errors <- numeric(length(sample_sizes))
  erm_errors <- numeric(length(sample_sizes))

  m_grid <- seq(0.0, 2.0, length.out = 41)
  s_grid <- seq(0.5, 1.5, length.out = 41)

  for (i in seq_along(sample_sizes)) {
    n <- sample_sizes[i]
    m_errs <- numeric(10)
    e_errs <- numeric(10)

    for (j in 1:10) {
      data <- generate_folded_normal(true_mu, true_sigma, n)

      mom_params <- moment_estimator_fn(data)
      m_errs[j] <- orbit_dist_fn(mom_params[1], mom_params[2], true_mu, true_sigma)

      erm_params <- net_erm_fn(data, m_grid, s_grid)
      e_errs[j] <- orbit_dist_fn(erm_params[1], erm_params[2], true_mu, true_sigma)
    }

    moment_errors[i] <- mean(m_errs)
    erm_errors[i] <- mean(e_errs)
  }

  pdf("experiments/folded_normal_r.pdf", width=6, height=4)
  plot(sample_sizes, moment_errors, type="b", col="blue", pch=19, ylim=c(0, max(c(moment_errors, erm_errors))),
       xlab="Sample Size (n)", ylab=expression("Orbit Error " * d[G]), main="Folded Normal: Estimator Convergence")
  lines(sample_sizes, erm_errors, type="b", col="red", pch=15)
  legend("topright", legend=c("Moment Estimator", "Net-ERM"), col=c("blue", "red"), pch=c(19, 15))
  grid()
  dev.off()
}

# --- GMM Experiment ---
generate_gmm <- function(w, mu, sigma, n) {
  components <- sample(1:length(w), size=n, replace=TRUE, prob=w)
  data <- numeric(n)
  for (i in 1:length(w)) {
    idx <- (components == i)
    data[idx] <- rnorm(sum(idx), mean=mu[i], sd=sigma[i])
  }
  return(data)
}

orbit_dist_gmm <- function(w1, mu1, sig1, w2, mu2, sig2) {
  # k=2
  perm1 <- c(1, 2)
  perm2 <- c(2, 1)

  dist1 <- sum(abs(w1 - w2[perm1])) + max(abs(mu1 - mu2[perm1])) + max(abs(sig1 - sig2[perm1]))
  dist2 <- sum(abs(w1 - w2[perm2])) + max(abs(mu1 - mu2[perm2])) + max(abs(sig1 - sig2[perm2]))

  return(min(dist1, dist2))
}

run_gmm_experiments <- function() {
  true_w <- c(0.4, 0.6)
  true_mu <- c(-2.0, 2.0)
  true_sig <- c(1.0, 1.0)

  sample_sizes <- c(100, 500, 1000, 2000, 5000, 10000)
  em_errors <- numeric(length(sample_sizes))

  for (i in seq_along(sample_sizes)) {
    n <- sample_sizes[i]
    e_errs <- numeric(10)

    for (j in 1:10) {
      data <- generate_gmm(true_w, true_mu, true_sig, n)

      # EM algorithm
      fit <- Mclust(data, G=2, modelNames="V", verbose=FALSE)

      # Mclust returns parameters ordered by components.
      # fit$parameters$pro = weights
      # fit$parameters$mean = means
      # sqrt(fit$parameters$variance$sigmasq) = standard deviations
      w_hat <- fit$parameters$pro
      mu_hat <- as.vector(fit$parameters$mean)

      # In 'V' model for 1D, sigmasq is a vector
      sig_hat <- sqrt(fit$parameters$variance$sigmasq)
      if (length(sig_hat) == 1) {
          sig_hat <- rep(sig_hat, 2)
      }

      e_errs[j] <- orbit_dist_gmm(w_hat, mu_hat, sig_hat, true_w, true_mu, true_sig)
    }

    em_errors[i] <- mean(e_errs)
  }

  pdf("experiments/gmm_r.pdf", width=6, height=4)
  plot(sample_sizes, em_errors, type="b", col="darkgreen", pch=17,
       xlab="Sample Size (n)", ylab=expression("Orbit Error " * d[G]), main="Gaussian Mixture (k=2): EM Convergence")
  legend("topright", legend=c("EM Algorithm"), col=c("darkgreen"), pch=c(17))
  grid()
  dev.off()
}

run_folded_normal_experiments()
run_gmm_experiments()
