# ============================================================
# L-FN2 (MSL paper): Folded normal net-ERM at coarse resolution
#   Model: Y = |N(mu*, sigma*^2)|
#   Estimator: net-ERM on quotient chart mu in [0,M_m], sigma in [sigma_min, sigma_max]
#   Loss: negative log-likelihood of folded normal (stable log-cosh implementation)
#   Error: orbit metric d_G for sign action (mu identified up to sign)
#
# PURPOSE (L-FN2):
#   - Show net-ERM is implementable on modest nets (gamma = eps, eps/2).
#   - Empirically validate orbit-PAC behavior qualitatively.
#   - Report runtime scaling vs n*|Gamma|.
#
# SAVES to:
#   E:\msl paper\exp_L_FN2_outputs\
# ============================================================

# ----------------------------
# 0) Packages + reproducibility
# ----------------------------
suppressPackageStartupMessages({
  library(ggplot2)
})

RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
options(stringsAsFactors = FALSE)

MASTER_SEED <- 20260228L
set.seed(MASTER_SEED)

# ----------------------------
# 1) Save paths (Windows)
# ----------------------------
BASE_DIR <- "E:/msl paper"   # <-- your target folder
out_dir  <- file.path(BASE_DIR, "exp_L_FN2_outputs")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ----------------------------
# 2) Compute tier: laptop vs server
# ----------------------------
# laptop:
#   - moderate reps, single core, coarse n grid -> finishes fast
# server:
#   - larger reps, optional parallel -> paper-quality Monte Carlo + bigger nets
COMPUTE_TIER <- "laptop"  # set to "server" for full runs

if (COMPUTE_TIER == "laptop") {
  R_reps <- 150L
  N_CORES <- 1L
  n_grid <- c(50L, 100L, 200L, 400L, 800L)
} else {
  R_reps <- 800L
  N_CORES <- 1L  # keep 1 by default on Windows; raise if you enable parallel socket clusters
  n_grid <- c(25L, 50L, 100L, 200L, 400L, 800L, 1600L)
}

# ----------------------------
# 3) Sieve + theta* grid
# ----------------------------
M_m <- 3.0
sigma_min <- 0.5
sigma_max <- 2.0

mu_grid_star    <- c(0.10 * M_m, 0.25 * M_m, 0.50 * M_m, 0.75 * M_m, 1.00 * M_m)
sigma_grid_star <- c(sigma_min, (sigma_min + sigma_max) / 2, sigma_max)

theta_star <- expand.grid(mu = mu_grid_star, sigma = sigma_grid_star)
theta_star$theta_id <- seq_len(nrow(theta_star))
theta_star$facet_lab <- sprintf("id=%d (mu*=%.3g, sigma*=%.3g)",
                                theta_star$theta_id, theta_star$mu, theta_star$sigma)

# ----------------------------
# 4) Fixed epsilon for L-FN2 + gamma choices
# ----------------------------
# L-FN2 spec: gamma in {eps, eps/2} under fixed eps
EPS_TARGET <- 0.10
gamma_grid <- c(EPS_TARGET, EPS_TARGET / 2)

# ----------------------------
# 5) Numerically stable folded-normal loss
# ----------------------------
log_cosh_stable <- function(t) {
  # log(cosh(t)) computed stably:
  # cosh(t) = exp(a)/2 * (1 + exp(-2a)), a = |t|
  a <- abs(t)
  a - log(2) + log1p(exp(-2 * a))
}

folded_nll_vec <- function(y, mu, sigma) {
  # y: numeric vector >=0
  # mu>=0 in quotient chart
  # returns vector of negative log-likelihood values at each y
  s2 <- sigma * sigma
  a <- (mu * y) / s2
  # From: loss = log(σ√(2π)/2) + (y^2+mu^2)/(2σ^2) - log cosh(mu y / σ^2)
  # Using stable log-cosh:
  log(sigma * sqrt(2 * pi) / 2) + (y*y + mu*mu) / (2 * s2) - log_cosh_stable(a)
}

dG_sign <- function(mu_hat, sigma_hat, mu_star, sigma_star) {
  # output mu_hat is in quotient chart (>=0); compare to |mu_star|
  sqrt((sigma_hat - sigma_star)^2 + (mu_hat - abs(mu_star))^2)
}

seed_for_config <- function(master, theta_id, n, gamma_index) {
  as.integer((master + 100000L * theta_id + 1000L * as.integer(n) + 7L * gamma_index) %% .Machine$integer.max)
}

# ----------------------------
# 6) Build quotient grid-net (Gamma_m) for given gamma
# ----------------------------
grid_net_quotient <- function(M_m, sigma_min, sigma_max, gamma) {
  mu_seq <- seq(0, M_m, by = gamma)
  if (tail(mu_seq, 1) < M_m) mu_seq <- c(mu_seq, M_m)
  
  sg_seq <- seq(sigma_min, sigma_max, by = gamma)
  if (tail(sg_seq, 1) < sigma_max) sg_seq <- c(sg_seq, sigma_max)
  
  grid <- expand.grid(mu = mu_seq, sigma = sg_seq)
  grid
}

# ----------------------------
# 7) Net-ERM on a fixed net
# ----------------------------
neterm_folded <- function(y, net_df) {
  # y: vector length n
  # net_df: data.frame with columns mu, sigma
  # returns list(mu_hat, sigma_hat, min_risk, elapsed_eval)
  # Timing includes the full evaluation over the net (dominant term).
  t0 <- proc.time()[["elapsed"]]
  
  # Evaluate empirical risk for each candidate
  # (loop is simple and robust; net is intentionally modest in L-FN2)
  n_cand <- nrow(net_df)
  risks <- numeric(n_cand)
  
  for (j in seq_len(n_cand)) {
    mu <- net_df$mu[j]
    sg <- net_df$sigma[j]
    risks[j] <- mean(folded_nll_vec(y, mu, sg))
  }
  
  j_star <- which.min(risks)
  
  t1 <- proc.time()[["elapsed"]]
  list(
    mu_hat = net_df$mu[j_star],
    sigma_hat = net_df$sigma[j_star],
    min_risk = risks[j_star],
    elapsed_eval = t1 - t0,
    net_size = n_cand
  )
}

# ----------------------------
# 8) Simulation: store quantiles, success, runtime
# ----------------------------
quantile_probs <- c(0.10, 0.25, 0.50, 0.75, 0.90)

out_quantiles <- list()
out_success   <- list()
out_runtime   <- list()

idx_q <- 0L
idx_s <- 0L
idx_t <- 0L

for (row in seq_len(nrow(theta_star))) {
  mu_star <- theta_star$mu[row]
  sg_star <- theta_star$sigma[row]
  tid     <- theta_star$theta_id[row]
  flab    <- theta_star$facet_lab[row]
  
  for (gidx in seq_along(gamma_grid)) {
    gamma <- gamma_grid[gidx]
    gamma_lab <- sprintf("gamma=%.3g", gamma)
    
    # Build net once per (theta*, gamma) and reuse across n and reps
    net_df <- grid_net_quotient(M_m, sigma_min, sigma_max, gamma)
    net_size <- nrow(net_df)
    
    for (n in n_grid) {
      set.seed(seed_for_config(MASTER_SEED, tid, n, gidx))
      
      errs <- numeric(R_reps)
      times <- numeric(R_reps)
      
      # Monte Carlo reps
      for (r in seq_len(R_reps)) {
        # deterministic per-rep seed (stable under reruns)
        set.seed(as.integer(seed_for_config(MASTER_SEED, tid, n, gidx) + 1000000L * r))
        
        X <- rnorm(n, mean = mu_star, sd = sg_star)
        Y <- abs(X)
        
        fit <- neterm_folded(Y, net_df)
        
        errs[r] <- dG_sign(fit$mu_hat, fit$sigma_hat, mu_star, sg_star)
        times[r] <- fit$elapsed_eval
      }
      
      # Quantiles of orbit error
      qs <- as.numeric(stats::quantile(errs, probs = quantile_probs, names = FALSE, type = 7))
      idx_q <- idx_q + 1L
      out_quantiles[[idx_q]] <- data.frame(
        model = "folded_normal",
        exp_id = "L-FN2",
        theta_id = tid,
        facet_lab = flab,
        mu_star = mu_star,
        sigma_star = sg_star,
        n = as.integer(n),
        gamma = gamma,
        gamma_lab = gamma_lab,
        net_size = as.integer(net_size),
        q10 = qs[1],
        q25 = qs[2],
        q50 = qs[3],
        q75 = qs[4],
        q90 = qs[5]
      )
      
      # Success probability at EPS_TARGET
      idx_s <- idx_s + 1L
      out_success[[idx_s]] <- data.frame(
        model = "folded_normal",
        exp_id = "L-FN2",
        theta_id = tid,
        facet_lab = flab,
        mu_star = mu_star,
        sigma_star = sg_star,
        n = as.integer(n),
        eps = EPS_TARGET,
        gamma = gamma,
        gamma_lab = gamma_lab,
        net_size = as.integer(net_size),
        p_success = mean(errs <= EPS_TARGET)
      )
      
      # Runtime summary vs n*|Gamma|
      idx_t <- idx_t + 1L
      out_runtime[[idx_t]] <- data.frame(
        model = "folded_normal",
        exp_id = "L-FN2",
        theta_id = tid,
        facet_lab = flab,
        mu_star = mu_star,
        sigma_star = sg_star,
        n = as.integer(n),
        gamma = gamma,
        gamma_lab = gamma_lab,
        net_size = as.integer(net_size),
        n_times_net = as.numeric(n) * as.numeric(net_size),
        time_med = stats::median(times),
        time_mean = mean(times),
        time_q90 = as.numeric(stats::quantile(times, 0.90, names = FALSE)),
        throughput_med = (as.numeric(n) * as.numeric(net_size)) / stats::median(times)
      )
    }
  }
}

df_quantiles <- do.call(rbind, out_quantiles)
df_success   <- do.call(rbind, out_success)
df_runtime   <- do.call(rbind, out_runtime)

# ----------------------------
# 9) Save CSVs
# ----------------------------
write.csv(df_quantiles, file = file.path(out_dir, "L_FN2_quantiles.csv"), row.names = FALSE)
write.csv(df_success,   file = file.path(out_dir, "L_FN2_success.csv"),   row.names = FALSE)
write.csv(df_runtime,   file = file.path(out_dir, "L_FN2_runtime.csv"),   row.names = FALSE)

# ----------------------------
# 10) Plots (ggplot2)
# ----------------------------

# (A) Orbit error ribbons vs n, comparing gamma
p_err <- ggplot(df_quantiles, aes(x = n, group = gamma_lab)) +
  geom_ribbon(aes(ymin = q25, ymax = q75, fill = gamma_lab), alpha = 0.18) +
  geom_line(aes(y = q50, color = gamma_lab), linewidth = 0.55) +
  geom_point(aes(y = q50, color = gamma_lab), size = 1.2) +
  scale_x_log10(breaks = n_grid) +
  labs(
    title = "L-FN2: Net-ERM (coarse nets) — orbit error vs n",
    subtitle = sprintf("Compare gamma ∈ {eps, eps/2} with eps = %.2f; ribbons show IQR (q25–q75).", EPS_TARGET),
    x = "n (log scale)",
    y = expression(d[G] * "(hat(theta), theta*)"),
    color = "net resolution",
    fill  = "net resolution"
  ) +
  facet_wrap(~ facet_lab, scales = "free_y", ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(out_dir, "L_FN2_error_ribbons_vs_n.pdf"),
  plot = p_err,
  width = 10, height = 7, device = "pdf"
)

# (B) Success probability vs n at eps=EPS_TARGET, comparing gamma
p_succ <- ggplot(df_success, aes(x = n, y = p_success, color = gamma_lab, group = gamma_lab)) +
  geom_hline(yintercept = 0.90, linetype = 2, linewidth = 0.35) +
  geom_hline(yintercept = 0.95, linetype = 3, linewidth = 0.35) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  scale_x_log10(breaks = n_grid) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "L-FN2: Net-ERM — success probability vs n",
    subtitle = sprintf("Success = 1{ d_G ≤ eps }, eps = %.2f. Reference: 0.90 (delta=0.10), 0.95 (delta=0.05).", EPS_TARGET),
    x = "n (log scale)",
    y = expression(P(d[G] <= epsilon)),
    color = "net resolution"
  ) +
  facet_wrap(~ facet_lab, ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(out_dir, "L_FN2_success_vs_n.pdf"),
  plot = p_succ,
  width = 10, height = 7, device = "pdf"
)

# (C) Runtime scaling: time_med vs n*|Gamma|, comparing gamma
p_time <- ggplot(df_runtime, aes(x = n_times_net, y = time_med, color = gamma_lab, group = gamma_lab)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    title = "L-FN2: Net-ERM — runtime scaling",
    subtitle = "Median elapsed time per replicate vs n·|Gamma| (both log scales).",
    x = expression(n %*% "|" * Gamma[m] * "|"),
    y = "median time per replicate (seconds, log scale)",
    color = "net resolution"
  ) +
  facet_wrap(~ facet_lab, ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(out_dir, "L_FN2_runtime_vs_n_times_net.pdf"),
  plot = p_time,
  width = 10, height = 7, device = "pdf"
)

# ----------------------------
# 11) Session info
# ----------------------------
sink(file.path(out_dir, "sessionInfo.txt"))
cat("MASTER_SEED =", MASTER_SEED, "\n")
cat("COMPUTE_TIER =", COMPUTE_TIER, "\n")
cat("R_reps =", R_reps, "\n")
cat("EPS_TARGET =", EPS_TARGET, "\n")
cat("gamma_grid =", paste(gamma_grid, collapse = ", "), "\n")
cat("Sieve: M_m =", M_m, "; sigma in [", sigma_min, ",", sigma_max, "]\n\n", sep = "")
print(sessionInfo())
sink()

message("Done. Outputs written to: ", out_dir)
