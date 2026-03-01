# ============================================================
# L-FN3 (MSL paper): τ-approximate net-ERM sensitivity (folded normal)
#   Model: Y = |N(mu*, sigma*^2)|
#   Candidate set: fixed quotient grid net Θ_{m,γ}  (same net for all τ)
#   Exact objective: R_n(θ) = mean( -log p_{θ}(Y_i) )
#   Approximate ERM: pick any θ with R_n(θ) <= R_n^* + τ
#   Output: success probability p̂_{n,ε} versus τ for several n
#
# SAVES to:
#   E:\msl paper\exp_L_FN3_outputs\
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
BASE_DIR <- "E:/msl paper"
out_dir  <- file.path(BASE_DIR, "exp_L_FN3_outputs")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ----------------------------
# 2) Compute tier: laptop vs server
# ----------------------------
COMPUTE_TIER <- "laptop"  # "server" for heavier runs

if (COMPUTE_TIER == "laptop") {
  R_reps <- 150L
  n_grid <- c(100L, 200L, 400L, 800L)
} else {
  R_reps <- 800L
  n_grid <- c(50L, 100L, 200L, 400L, 800L, 1600L)
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
# 4) Fixed (ε, γ) and τ-grid
# ----------------------------
# We keep candidate set fixed: choose a single coarse γ (implementable)
EPS_TARGET <- 0.10
GAMMA_FIXED <- EPS_TARGET / 2  # typical choice; do NOT change across τ

# τ is additive tolerance in empirical risk units (negative log-lik)
# Use a log-spaced grid plus τ=0
tau_grid <- c(0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2)

# ----------------------------
# 5) Numerically stable folded-normal loss
# ----------------------------
log_cosh_stable <- function(t) {
  a <- abs(t)
  a - log(2) + log1p(exp(-2 * a))
}

folded_nll_vec <- function(y, mu, sigma) {
  s2 <- sigma * sigma
  a <- (mu * y) / s2
  log(sigma * sqrt(2 * pi) / 2) + (y*y + mu*mu) / (2 * s2) - log_cosh_stable(a)
}

dG_sign <- function(mu_hat, sigma_hat, mu_star, sigma_star) {
  sqrt((sigma_hat - sigma_star)^2 + (mu_hat - abs(mu_star))^2)
}

seed_for_config <- function(master, theta_id, n) {
  as.integer((master + 100000L * theta_id + 1000L * as.integer(n)) %% .Machine$integer.max)
}

# ----------------------------
# 6) Build quotient grid-net (Γ_m) at fixed γ
# ----------------------------
grid_net_quotient <- function(M_m, sigma_min, sigma_max, gamma) {
  mu_seq <- seq(0, M_m, by = gamma)
  if (tail(mu_seq, 1) < M_m) mu_seq <- c(mu_seq, M_m)
  
  sg_seq <- seq(sigma_min, sigma_max, by = gamma)
  if (tail(sg_seq, 1) < sigma_max) sg_seq <- c(sg_seq, sigma_max)
  
  expand.grid(mu = mu_seq, sigma = sg_seq)
}

NET_DF <- grid_net_quotient(M_m, sigma_min, sigma_max, GAMMA_FIXED)
NET_SIZE <- nrow(NET_DF)

# ----------------------------
# 7) Exact risks over net + τ-approx selection
# ----------------------------
net_risks <- function(y, net_df) {
  n_cand <- nrow(net_df)
  risks <- numeric(n_cand)
  for (j in seq_len(n_cand)) {
    risks[j] <- mean(folded_nll_vec(y, net_df$mu[j], net_df$sigma[j]))
  }
  risks
}

tau_select_index <- function(risks, tau, tie_rule = c("uniform", "first")) {
  tie_rule <- match.arg(tie_rule)
  rmin <- min(risks)
  ok <- which(risks <= rmin + tau)
  if (length(ok) == 0L) stop("No feasible index under tau (should not happen).")
  if (tie_rule == "first") return(ok[1])
  # uniform among feasible candidates (models 'optimization noise' within tolerance)
  sample(ok, size = 1)
}

# ----------------------------
# 8) Simulation: success vs τ, for several n
# ----------------------------
out_tau <- list()
idx <- 0L

# choose tie rule: "uniform" is the more honest stress test
TIE_RULE <- "uniform"  # or "first"

for (row in seq_len(nrow(theta_star))) {
  mu_star <- theta_star$mu[row]
  sg_star <- theta_star$sigma[row]
  tid     <- theta_star$theta_id[row]
  flab    <- theta_star$facet_lab[row]
  
  for (n in n_grid) {
    set.seed(seed_for_config(MASTER_SEED, tid, n))
    
    # For each τ, accumulate successes across reps
    succ_mat <- matrix(0, nrow = R_reps, ncol = length(tau_grid))
    
    # runtime (optional): cost dominated by risk evaluation over net
    time_eval <- numeric(R_reps)
    
    for (r in seq_len(R_reps)) {
      set.seed(as.integer(seed_for_config(MASTER_SEED, tid, n) + 1000000L * r))
      X <- rnorm(n, mean = mu_star, sd = sg_star)
      Y <- abs(X)
      
      t0 <- proc.time()[["elapsed"]]
      risks <- net_risks(Y, NET_DF)
      t1 <- proc.time()[["elapsed"]]
      time_eval[r] <- (t1 - t0)
      
      for (k in seq_along(tau_grid)) {
        tau <- tau_grid[k]
        j <- tau_select_index(risks, tau = tau, tie_rule = TIE_RULE)
        mu_hat <- NET_DF$mu[j]
        sg_hat <- NET_DF$sigma[j]
        err <- dG_sign(mu_hat, sg_hat, mu_star, sg_star)
        succ_mat[r, k] <- as.integer(err <= EPS_TARGET)
      }
    }
    
    # summarize across reps for each τ
    for (k in seq_along(tau_grid)) {
      idx <- idx + 1L
      out_tau[[idx]] <- data.frame(
        model = "folded_normal",
        exp_id = "L-FN3",
        theta_id = tid,
        facet_lab = flab,
        mu_star = mu_star,
        sigma_star = sg_star,
        n = as.integer(n),
        eps = EPS_TARGET,
        gamma = GAMMA_FIXED,
        net_size = as.integer(NET_SIZE),
        tau = tau_grid[k],
        tie_rule = TIE_RULE,
        p_success = mean(succ_mat[, k]),
        p_se = sqrt(mean(succ_mat[, k]) * (1 - mean(succ_mat[, k])) / R_reps),
        time_eval_med = stats::median(time_eval),
        time_eval_mean = mean(time_eval)
      )
    }
  }
}

df_tau <- do.call(rbind, out_tau)

# ----------------------------
# 9) Save CSV
# ----------------------------
write.csv(df_tau, file = file.path(out_dir, "L_FN3_success_vs_tau.csv"), row.names = FALSE)

# ----------------------------
# 10) Plots (ggplot2)
# ----------------------------

# Plot A: success vs τ (log x), separate curves by n
p_tau <- ggplot(df_tau, aes(x = tau, y = p_success, group = factor(n), color = factor(n))) +
  geom_hline(yintercept = 0.90, linetype = 2, linewidth = 0.35) +
  geom_hline(yintercept = 0.95, linetype = 3, linewidth = 0.35) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  # τ includes 0; show on pseudo-log scale by mapping 0 -> min positive for plotting
  scale_x_continuous(
    trans = "log10",
    breaks = tau_grid[tau_grid > 0],
    labels = format(tau_grid[tau_grid > 0], scientific = TRUE)
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "L-FN3: τ-approximate net-ERM — success probability vs τ",
    subtitle = sprintf("Fixed net (gamma=%.3g, |Gamma|=%d), eps=%.2f, tie rule: %s.",
                       GAMMA_FIXED, NET_SIZE, EPS_TARGET, TIE_RULE),
    x = expression(tau * "  (log scale; τ=0 omitted on log axis)"),
    y = expression(P(d[G] <= epsilon)),
    color = "n"
  ) +
  facet_wrap(~ facet_lab, ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

# To avoid log(0) issues, plot τ=0 separately as a point at τ=minpos/10
minpos <- min(tau_grid[tau_grid > 0])
df0 <- df_tau[df_tau$tau == 0, ]
df0$tau_plot <- minpos / 10
dfp <- df_tau[df_tau$tau > 0, ]
dfp$tau_plot <- dfp$tau

p_tau2 <- ggplot() +
  geom_hline(yintercept = 0.90, linetype = 2, linewidth = 0.35) +
  geom_hline(yintercept = 0.95, linetype = 3, linewidth = 0.35) +
  geom_line(data = dfp, aes(x = tau_plot, y = p_success, group = factor(n), color = factor(n)), linewidth = 0.6) +
  geom_point(data = dfp, aes(x = tau_plot, y = p_success, color = factor(n)), size = 1.2) +
  geom_point(data = df0, aes(x = tau_plot, y = p_success, color = factor(n)), shape = 21, fill = "white", size = 2.0) +
  scale_x_log10(
    breaks = c(minpos/10, tau_grid[tau_grid > 0]),
    labels = c("0", format(tau_grid[tau_grid > 0], scientific = TRUE))
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "L-FN3: τ-approximate net-ERM — success probability vs τ",
    subtitle = sprintf("Fixed net (gamma=%.3g, |Gamma|=%d), eps=%.2f, tie rule: %s. Open circles at τ=0.",
                       GAMMA_FIXED, NET_SIZE, EPS_TARGET, TIE_RULE),
    x = expression(tau),
    y = expression(P(d[G] <= epsilon)),
    color = "n"
  ) +
  facet_wrap(~ facet_lab, ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(out_dir, "L_FN3_success_vs_tau.pdf"),
  plot = p_tau2,
  width = 10, height = 7, device = "pdf"
)

# Plot B: time vs n (median net evaluation time)
df_time <- unique(df_tau[, c("theta_id","facet_lab","n","gamma","net_size","time_eval_med")])
p_time <- ggplot(df_time, aes(x = n, y = time_eval_med)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_x_log10(breaks = n_grid) +
  scale_y_log10() +
  labs(
    title = "L-FN3: runtime for fixed-net risk evaluation",
    subtitle = sprintf("gamma=%.3g, |Gamma|=%d. Median time per replicate (log scale).", GAMMA_FIXED, NET_SIZE),
    x = "n (log scale)",
    y = "median time per replicate (seconds, log scale)"
  ) +
  facet_wrap(~ facet_lab, ncol = 3) +
  theme_bw(base_size = 11) +
  theme(
    strip.text = element_text(size = 9),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(out_dir, "L_FN3_runtime_vs_n.pdf"),
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
cat("GAMMA_FIXED =", GAMMA_FIXED, "\n")
cat("NET_SIZE =", NET_SIZE, "\n")
cat("tau_grid =", paste(tau_grid, collapse = ", "), "\n")
cat("TIE_RULE =", TIE_RULE, "\n")
cat("Sieve: M_m =", M_m, "; sigma in [", sigma_min, ",", sigma_max, "]\n\n", sep = "")
print(sessionInfo())
sink()

message("Done. Outputs written to: ", out_dir)
