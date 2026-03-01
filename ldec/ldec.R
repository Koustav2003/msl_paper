# ============================================================
# L-DEC (MSL paper): Improper-to-proper decoding on a modest net
# FIXED: avoids log10(-Inf) in runtime plots by flooring timings > 0
# SAVES to: E:\msl paper\exp_L_DEC_outputs\
# ============================================================

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
BASE_DIR <- "E:/msl paper"
out_dir  <- file.path(BASE_DIR, "exp_L_DEC_outputs")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ----------------------------
# 2) Compute tier: laptop vs server
# ----------------------------
COMPUTE_TIER <- "laptop"  # set to "server" for heavier sweeps

if (COMPUTE_TIER == "laptop") {
  R_reps <- 120L
  n_grid <- c(100L, 300L, 800L)
  gamma_grid <- c(0.40, 0.20, 0.10)
  GRID_N <- 1200L
} else {
  R_reps <- 600L
  n_grid <- c(50L, 100L, 300L, 800L, 1600L)
  gamma_grid <- c(0.60, 0.40, 0.30, 0.20, 0.15, 0.10)
  GRID_N <- 2400L
}

# ----------------------------
# 3) Sieve (folded normal) + θ* grid
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
# 4) Numerical grid for d_stat on [0, y_max]
# ----------------------------
y_max <- M_m + 8 * sigma_max
y_grid <- seq(0, y_max, length.out = GRID_N)

trapz <- function(x, f) {
  sum((x[-1] - x[-length(x)]) * (f[-1] + f[-length(f)]) / 2)
}

# ----------------------------
# 5) Folded normal density on y>=0 (stable log-sum-exp)
# ----------------------------
folded_density_vec <- function(y, mu, sigma) {
  log1 <- dnorm(y, mean = mu,  sd = sigma, log = TRUE)
  log2 <- dnorm(y, mean = -mu, sd = sigma, log = TRUE)
  m <- pmax(log1, log2)
  logsum <- m + log(exp(log1 - m) + exp(log2 - m))
  exp(logsum)
}

normalize_density_on_grid <- function(p, y_grid) {
  p <- pmax(p, 0)
  Z <- trapz(y_grid, p)
  if (!is.finite(Z) || Z <= 0) stop("Normalization failed (nonpositive integral).")
  p / Z
}

hellinger2_grid <- function(p, q, y_grid) {
  ip <- trapz(y_grid, sqrt(p * q))
  out <- 1 - ip
  max(0, min(1, out))
}

# Orbit metric for sign symmetry, with canonical μ>=0
dG_sign <- function(mu_hat, sigma_hat, mu_star, sigma_star) {
  sqrt((sigma_hat - sigma_star)^2 + (mu_hat - abs(mu_star))^2)
}

# ----------------------------
# 6) Modest net Θ_{m,γ} on quotient chart
# ----------------------------
grid_net_quotient <- function(M_m, sigma_min, sigma_max, gamma) {
  mu_seq <- seq(0, M_m, by = gamma)
  if (tail(mu_seq, 1) < M_m) mu_seq <- c(mu_seq, M_m)
  
  sg_seq <- seq(sigma_min, sigma_max, by = gamma)
  if (tail(sg_seq, 1) < sigma_max) sg_seq <- c(sg_seq, sigma_max)
  
  expand.grid(mu = mu_seq, sigma = sg_seq)
}

# ----------------------------
# 7) Improper learner: KDE -> Q-hat density on y_grid
# ----------------------------
kde_qhat_on_grid <- function(y_samples, y_grid) {
  d <- stats::density(
    y_samples,
    from = min(y_grid),
    to   = max(y_grid),
    n    = length(y_grid),
    cut  = 0
  )
  q <- d$y
  normalize_density_on_grid(q, y_grid)
}

# ----------------------------
# 8) Decoder: argmin_{θ in Θ} d_stat(Q-hat, P_θ)
#    FIX: floor timing to avoid zeros (log10 issues)
# ----------------------------
PRECOMP_MAX_CAND <- 4000L
TIME_FLOOR <- 1e-6  # seconds; ensures strictly positive times

precompute_Pmat <- function(net_df, y_grid) {
  n_cand <- nrow(net_df)
  Pmat <- matrix(NA_real_, nrow = n_cand, ncol = length(y_grid))
  for (j in seq_len(n_cand)) {
    p <- folded_density_vec(y_grid, net_df$mu[j], net_df$sigma[j])
    Pmat[j, ] <- normalize_density_on_grid(p, y_grid)
  }
  Pmat
}

decode_argmin_hellinger2 <- function(q_hat, net_df, y_grid, Pmat = NULL) {
  t0 <- proc.time()[["elapsed"]]
  
  n_cand <- nrow(net_df)
  d2 <- numeric(n_cand)
  
  if (!is.null(Pmat)) {
    for (j in seq_len(n_cand)) d2[j] <- hellinger2_grid(Pmat[j, ], q_hat, y_grid)
  } else {
    for (j in seq_len(n_cand)) {
      p <- folded_density_vec(y_grid, net_df$mu[j], net_df$sigma[j])
      p <- normalize_density_on_grid(p, y_grid)
      d2[j] <- hellinger2_grid(p, q_hat, y_grid)
    }
  }
  
  j_star <- which.min(d2)
  t1 <- proc.time()[["elapsed"]]
  
  elapsed <- t1 - t0
  if (!is.finite(elapsed) || elapsed <= 0) elapsed <- TIME_FLOOR
  elapsed <- max(elapsed, TIME_FLOOR)
  
  list(j_star = j_star, d2_min = d2[j_star], elapsed_decode = elapsed)
}

seed_for_config <- function(master, theta_id, n, gamma_index, rep) {
  as.integer((master +
                100000L  * theta_id +
                1000L    * as.integer(n) +
                17L      * gamma_index +
                1000000L * rep) %% .Machine$integer.max)
}

# ----------------------------
# 9) Main experiment loop
# ----------------------------
trial_rows <- list()
rt_rows <- list()
idx <- 0L
idx_rt <- 0L

for (gidx in seq_along(gamma_grid)) {
  gamma <- gamma_grid[gidx]
  gamma_lab <- sprintf("gamma=%.3g", gamma)
  
  net_df <- grid_net_quotient(M_m, sigma_min, sigma_max, gamma)
  net_size <- nrow(net_df)
  
  Pmat <- NULL
  if (net_size <= PRECOMP_MAX_CAND) Pmat <- precompute_Pmat(net_df, y_grid)
  
  decode_times <- numeric(0)
  
  for (row in seq_len(nrow(theta_star))) {
    mu_star <- theta_star$mu[row]
    sg_star <- theta_star$sigma[row]
    tid     <- theta_star$theta_id[row]
    flab    <- theta_star$facet_lab[row]
    
    p_star <- folded_density_vec(y_grid, mu_star, sg_star)
    p_star <- normalize_density_on_grid(p_star, y_grid)
    
    for (n in n_grid) {
      for (r in seq_len(R_reps)) {
        set.seed(seed_for_config(MASTER_SEED, tid, n, gidx, r))
        
        X <- rnorm(n, mean = mu_star, sd = sg_star)
        Y <- abs(X)
        
        q_hat <- kde_qhat_on_grid(Y, y_grid)
        a <- hellinger2_grid(q_hat, p_star, y_grid)
        
        dec <- decode_argmin_hellinger2(q_hat, net_df, y_grid, Pmat = Pmat)
        decode_times <- c(decode_times, dec$elapsed_decode)
        
        jhat <- dec$j_star
        mu_hat <- net_df$mu[jhat]
        sg_hat <- net_df$sigma[jhat]
        e <- dG_sign(mu_hat, sg_hat, mu_star, sg_star)
        
        idx <- idx + 1L
        trial_rows[[idx]] <- data.frame(
          model = "folded_normal",
          exp_id = "L-DEC",
          compute_tier = COMPUTE_TIER,
          theta_id = tid,
          facet_lab = flab,
          n = as.integer(n),
          gamma = gamma,
          gamma_lab = gamma_lab,
          net_size = as.integer(net_size),
          mu_star = mu_star,
          sigma_star = sg_star,
          mu_hat = mu_hat,
          sigma_hat = sg_hat,
          a_dstat = a,
          e_orbit = e,
          decode_time = dec$elapsed_decode
        )
      }
    }
  }
  
  # runtime summary (all strictly positive by construction)
  idx_rt <- idx_rt + 1L
  rt_rows[[idx_rt]] <- data.frame(
    model = "folded_normal",
    exp_id = "L-DEC",
    compute_tier = COMPUTE_TIER,
    gamma = gamma,
    gamma_lab = gamma_lab,
    net_size = as.integer(net_size),
    decode_time_med  = max(stats::median(decode_times), TIME_FLOOR),
    decode_time_mean = max(mean(decode_times), TIME_FLOOR),
    decode_time_q90  = max(as.numeric(stats::quantile(decode_times, 0.90, names = FALSE)), TIME_FLOOR)
  )
}

df_trials  <- do.call(rbind, trial_rows)
df_runtime <- do.call(rbind, rt_rows)

# ----------------------------
# 10) Save CSVs
# ----------------------------
write.csv(df_trials,  file = file.path(out_dir, "L_DEC_trials.csv"), row.names = FALSE)
write.csv(df_runtime, file = file.path(out_dir, "L_DEC_runtime_by_gamma.csv"), row.names = FALSE)

# ----------------------------
# 11) Plots (ggplot2)
# ----------------------------

# (A) Decoded orbit error vs distributional discrepancy
a_cap <- as.numeric(stats::quantile(df_trials$a_dstat, probs = 0.995, names = FALSE))
df_plot <- df_trials
df_plot$a_dstat_cap <- pmin(df_plot$a_dstat, a_cap)

BIN_COUNT <- 20L
breaks <- seq(0, a_cap, length.out = BIN_COUNT + 1L)
df_plot$abin <- cut(df_plot$a_dstat_cap, breaks = breaks, include.lowest = TRUE, right = FALSE)

summ_bin <- aggregate(
  e_orbit ~ gamma_lab + net_size + abin,
  data = df_plot,
  FUN = function(z) c(
    q25 = stats::quantile(z, 0.25, names = FALSE),
    q50 = stats::quantile(z, 0.50, names = FALSE),
    q75 = stats::quantile(z, 0.75, names = FALSE)
  )
)
summ_bin$q25 <- summ_bin$e_orbit[, "q25"]
summ_bin$q50 <- summ_bin$e_orbit[, "q50"]
summ_bin$q75 <- summ_bin$e_orbit[, "q75"]
summ_bin$e_orbit <- NULL

bin_levels <- levels(df_plot$abin)
bin_mids <- vapply(bin_levels, function(lbl) {
  nums <- gsub("\\[|\\)|\\]|\\(|\\s", "", lbl)
  parts <- strsplit(nums, ",", fixed = TRUE)[[1]]
  lo <- as.numeric(parts[1]); hi <- as.numeric(parts[2])
  0.5 * (lo + hi)
}, numeric(1))

summ_bin <- merge(summ_bin, data.frame(abin = bin_levels, a_mid = bin_mids), by = "abin", all.x = TRUE)

p_scatter <- ggplot(df_plot, aes(x = a_dstat_cap, y = e_orbit)) +
  geom_point(alpha = 0.22, size = 0.9) +
  geom_ribbon(
    data = summ_bin,
    aes(x = a_mid, ymin = q25, ymax = q75),
    inherit.aes = FALSE,
    alpha = 0.18
  ) +
  geom_line(
    data = summ_bin,
    aes(x = a_mid, y = q50),
    inherit.aes = FALSE,
    linewidth = 0.7
  ) +
  facet_wrap(~ gamma_lab, scales = "free_y") +
  labs(
    title = "L-DEC: decoded orbit error vs distributional discrepancy (modest nets)",
    subtitle = sprintf("d_stat = Hellinger^2 on [0, %.2f]; improper learner = KDE; ribbon=IQR, line=median (binned).", y_max),
    x = expression(a == H^2(Q[hat], P[theta^"*"])~~"(capped at 99.5%)"),
    y = expression(e == d[G](theta[hat], theta^"*"))
  ) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        legend.position = "none",
        strip.text = element_text(size = 9))

ggsave(
  filename = file.path(out_dir, "L_DEC_e_vs_a_scatter_quantiles.pdf"),
  plot = p_scatter,
  width = 10, height = 6.5, device = "pdf"
)

# (B) Runtime vs |Γ_m| (decode-only)
# FIX: runtime values already floored to TIME_FLOOR, but we also guard in plotting.
df_runtime_plot <- df_runtime
df_runtime_plot$decode_time_med_plot <- pmax(df_runtime_plot$decode_time_med, TIME_FLOOR)

p_rt <- ggplot(df_runtime_plot, aes(x = net_size, y = decode_time_med_plot)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.6) +
  scale_x_log10() +
  scale_y_log10() +
  labs(
    title = "L-DEC: decoding runtime vs net size",
    subtitle = "Median decoding time per replicate (log-log). KDE time excluded; this is pure d_stat-projection cost.",
    x = expression("|" * Gamma[m] * "|"),
    y = "median decode time (seconds, log scale)"
  ) +
  theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank())

ggsave(
  filename = file.path(out_dir, "L_DEC_runtime_vs_net_size.pdf"),
  plot = p_rt,
  width = 7.5, height = 5.2, device = "pdf"
)

# ----------------------------
# 12) Session info
# ----------------------------
sink(file.path(out_dir, "sessionInfo.txt"))
cat("MASTER_SEED =", MASTER_SEED, "\n")
cat("COMPUTE_TIER =", COMPUTE_TIER, "\n")
cat("R_reps =", R_reps, "\n")
cat("n_grid =", paste(n_grid, collapse = ", "), "\n")
cat("gamma_grid =", paste(gamma_grid, collapse = ", "), "\n")
cat("GRID_N =", GRID_N, "\n")
cat("y_max =", y_max, "\n")
cat("PRECOMP_MAX_CAND =", PRECOMP_MAX_CAND, "\n")
cat("TIME_FLOOR =", TIME_FLOOR, "\n")
cat("Sieve: M_m =", M_m, "; sigma in [", sigma_min, ",", sigma_max, "]\n\n", sep = "")
print(sessionInfo())
sink()

message("Done. Outputs written to: ", out_dir)
