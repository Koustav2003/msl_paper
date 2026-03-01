# ============================================================
# L_FN1_moment_baseline.R  (GGPlot2, no log10(0) warnings)
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
  library(ggplot2)
  library(scales)
})

# -----------------------
# USER PATH (Windows)
# -----------------------
out_dir <- "E:/msl paper"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------
# RNG + design
# -----------------------
set.seed(20260228)

R_reps  <- 500
n_grid  <- c(25, 50, 100, 200, 400, 800, 1600)

mu_grid    <- c(0.30, 0.75, 1.50, 2.25, 3.00)
sigma_grid <- c(0.50, 1.25, 2.00)

eps_grid <- c(0.05, 0.10, 0.20)

# -----------------------
# Helpers
# -----------------------

# Folded normal sample: Y = |mu + sigma Z|
r_folded <- function(n, mu, sigma) abs(rnorm(n, mean = mu, sd = sigma))

# Moment estimator from your definition (with (·)_+ protections)
mom_est <- function(y) {
  m2 <- mean(y^2)
  m4 <- mean(y^4)
  disc <- 3 * m2^2 - m4
  
  u_hat <- sqrt(pmax(disc, 0) / 2)          # u = mu^2 (estimated)
  v_raw <- m2 - u_hat                        # v = sigma^2 (raw)
  v_hat <- pmax(v_raw, 0)
  
  list(
    m2 = m2, m4 = m4, disc = disc,
    u_hat = u_hat, v_raw = v_raw, v_hat = v_hat,
    mu_hat = sqrt(pmax(u_hat, 0)),           # canonical rep mu_hat >= 0
    sigma_hat = sqrt(v_hat)
  )
}

# Orbit metric for sign group (safe even if you later allow mu_star < 0)
dG_sign <- function(mu_hat, sig_hat, mu_star, sig_star) {
  d1 <- sqrt((mu_hat - mu_star)^2 + (sig_hat - sig_star)^2)
  d2 <- sqrt((mu_hat + mu_star)^2 + (sig_hat - sig_star)^2)
  pmin(d1, d2)
}

quantile_summ <- function(x) {
  qs <- quantile(x, probs = c(0.10, 0.25, 0.50, 0.75, 0.90), names = FALSE, type = 7)
  tibble(q10 = qs[1], q25 = qs[2], q50 = qs[3], q75 = qs[4], q90 = qs[5])
}

# -----------------------
# Parameter grid
# -----------------------
theta_tbl <- tidyr::crossing(mu_star = mu_grid, sigma_star = sigma_grid) %>%
  arrange(sigma_star, mu_star) %>%
  mutate(theta_id = row_number(),
         facet_lab = sprintf("mu=%.2f, sigma=%.2f", mu_star, sigma_star))

# -----------------------
# Monte Carlo loop (vectorized over reps via map)
# -----------------------
one_setting <- function(mu_star, sigma_star, theta_id, facet_lab) {
  map_dfr(n_grid, function(n) {
    # simulate R reps
    reps <- map_dfr(seq_len(R_reps), function(r) {
      y <- r_folded(n, mu_star, sigma_star)
      est <- mom_est(y)
      
      err <- dG_sign(est$mu_hat, est$sigma_hat, mu_star, sigma_star)
      
      tibble(
        theta_id = theta_id,
        facet_lab = facet_lab,
        mu_star = mu_star,
        sigma_star = sigma_star,
        n = n,
        disc = est$disc,
        disc_neg = (est$disc < 0),
        v_raw = est$v_raw,
        v_neg = (est$v_raw < 0),
        sigma_zero = (est$v_hat <= 0),
        err = err
      )
    })
    reps
  })
}

raw <- pmap_dfr(theta_tbl, one_setting)

# -----------------------
# Summaries (CSV outputs)
# -----------------------

quantiles_df <- raw %>%
  group_by(theta_id, facet_lab, mu_star, sigma_star, n) %>%
  summarise(quantile_summ(err), .groups = "drop")

success_df <- raw %>%
  tidyr::crossing(eps = eps_grid) %>%
  group_by(theta_id, facet_lab, mu_star, sigma_star, n, eps) %>%
  summarise(p_success = mean(err <= eps), .groups = "drop")

diagnostics_df <- raw %>%
  group_by(theta_id, facet_lab, mu_star, sigma_star, n) %>%
  summarise(
    disc_neg_rate = mean(disc_neg),
    v_neg_rate = mean(v_neg),
    sigma_zero_rate = mean(sigma_zero),
    mean_err = mean(err),
    median_err = median(err),
    .groups = "drop"
  )

thresholds_df <- success_df %>%
  tidyr::crossing(delta = c(0.10, 0.05)) %>%
  mutate(target = 1 - delta) %>%
  group_by(theta_id, facet_lab, mu_star, sigma_star, eps, delta, target) %>%
  summarise(
    n_thr = {
      ok <- n[p_success >= target]
      if (length(ok) == 0) NA_real_ else min(ok)
    },
    .groups = "drop"
  )

# Write CSVs
write_csv(quantiles_df, file.path(out_dir, "L_FN1_quantiles.csv"))
write_csv(success_df,   file.path(out_dir, "L_FN1_success.csv"))
write_csv(diagnostics_df, file.path(out_dir, "L_FN1_diagnostics.csv"))
write_csv(thresholds_df,  file.path(out_dir, "L_FN1_thresholds.csv"))

# -----------------------
# PLOTS (PDF outputs)
# -----------------------

# (A) Error ribbons vs n (log-y is OK but guard against 0)
err_plot_df <- quantiles_df %>%
  mutate(across(c(q10, q25, q50, q75, q90), ~pmax(.x, 1e-12)))  # <-- prevents log10(0)

p_err <- ggplot(err_plot_df, aes(x = n)) +
  geom_ribbon(aes(ymin = q10, ymax = q90), alpha = 0.15) +
  geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.25) +
  geom_line(aes(y = q50), linewidth = 0.6) +
  scale_x_log10(breaks = n_grid) +
  scale_y_log10(labels = label_number(accuracy = 0.001)) +
  facet_wrap(~ facet_lab, scales = "free_y") +
  labs(x = "n (log scale)", y = "Orbit error d_G (log scale)",
       title = "L-FN1: Moment estimator orbit error quantiles vs n") +
  theme_bw(base_size = 11)

ggsave(file.path(out_dir, "L_FN1_error_ribbons_vs_n.pdf"), p_err, width = 12, height = 8)

# (B) Success vs n (DO NOT log y — it is a probability in [0,1])
p_succ <- ggplot(success_df, aes(x = n, y = p_success, group = factor(eps))) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_x_log10(breaks = n_grid) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0,1,0.2)) +
  facet_wrap(~ facet_lab) +
  labs(x = "n (log scale)", y = "Empirical success P(d_G <= eps)",
       title = "L-FN1: Success probability vs n",
       subtitle = "Moment estimator; success curves for eps ∈ {0.05, 0.10, 0.20}",
       color = "eps") +
  theme_bw(base_size = 11)

ggsave(file.path(out_dir, "L_FN1_success_vs_n_all_eps.pdf"), p_succ, width = 12, height = 8)

# (C) Diagnostics rates vs n (rates in [0,1] — keep linear y)
diag_long <- diagnostics_df %>%
  select(theta_id, facet_lab, mu_star, sigma_star, n,
         disc_neg_rate, v_neg_rate, sigma_zero_rate) %>%
  pivot_longer(cols = c(disc_neg_rate, v_neg_rate, sigma_zero_rate),
               names_to = "metric", values_to = "rate") %>%
  mutate(metric = recode(metric,
                         disc_neg_rate = "disc_neg_rate",
                         v_neg_rate = "v_neg_rate",
                         sigma_zero_rate = "sigma_zero_rate"))

p_diag <- ggplot(diag_long, aes(x = n, y = rate)) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.2) +
  scale_x_log10(breaks = n_grid) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0,1,0.2)) +
  facet_grid(metric ~ facet_lab) +
  labs(x = "n (log scale)", y = "Rate",
       title = "L-FN1: Moment inversion diagnostics vs n") +
  theme_bw(base_size = 10)

ggsave(file.path(out_dir, "L_FN1_diag_rates_vs_n.pdf"), p_diag, width = 14, height = 9)

# (D) Session info (reproducibility)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

message("Done. Wrote CSVs + PDFs to: ", out_dir)
