import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import itertools

import os
os.makedirs("experiments", exist_ok=True)

np.random.seed(42)

# --- Folded Normal Experiment ---
def generate_folded_normal(mu, sigma, n):
    x = np.random.normal(mu, sigma, n)
    return np.abs(x)

def orbit_dist_fn(mu1, sigma1, mu2, sigma2):
    d1 = np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)
    d2 = np.sqrt((mu1 + mu2)**2 + (sigma1 - sigma2)**2)
    return min(d1, d2)

def moment_estimator_fn(data):
    m2 = np.mean(data**2)
    m4 = np.mean(data**4)
    # Solve m2 = u + v, m4 = u^2 + 6uv + 3v^2
    # 3m2^2 - m4 = 2u^2
    val = (3 * m2**2 - m4) / 2
    u_hat = np.sqrt(max(0, val))
    v_hat = m2 - u_hat
    mu_hat = np.sqrt(max(0, u_hat))
    sigma_hat = np.sqrt(max(0, v_hat))
    return mu_hat, sigma_hat

def net_erm_fn(data, m_grid, s_grid):
    best_loss = np.inf
    best_params = (0, 1)
    for mu in m_grid:
        for sigma in s_grid:
            # negative log-likelihood
            # p(y) = 2/(sigma * sqrt(2pi)) * exp(-(y^2+mu^2)/(2sigma^2)) * cosh(mu y / sigma^2)
            # -log p(y) = log(sigma * sqrt(2pi) / 2) + (y^2+mu^2)/(2sigma^2) - log(cosh(mu y / sigma^2))

            # For numerical stability, log(cosh(x)) ~ |x| - log(2) for large x
            x = (mu * data) / (sigma**2)
            log_cosh = np.where(np.abs(x) < 50, np.log(np.cosh(x)), np.abs(x) - np.log(2.0))

            loss = np.mean(np.log(sigma * np.sqrt(2 * np.pi) / 2) + (data**2 + mu**2) / (2 * sigma**2) - log_cosh)
            if loss < best_loss:
                best_loss = loss
                best_params = (mu, sigma)
    return best_params

def run_folded_normal_experiments():
    true_mu = 1.0
    true_sigma = 1.0

    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    moment_errors = []
    erm_errors = []

    m_grid = np.linspace(0.0, 2.0, 41)
    s_grid = np.linspace(0.5, 1.5, 41)

    for n in sample_sizes:
        m_errs = []
        e_errs = []
        for _ in range(10):  # 10 trials
            data = generate_folded_normal(true_mu, true_sigma, n)

            # Moment estimator
            mu_mom, sig_mom = moment_estimator_fn(data)
            m_errs.append(orbit_dist_fn(mu_mom, sig_mom, true_mu, true_sigma))

            # ERM estimator
            mu_erm, sig_erm = net_erm_fn(data, m_grid, s_grid)
            e_errs.append(orbit_dist_fn(mu_erm, sig_erm, true_mu, true_sigma))

        moment_errors.append(np.mean(m_errs))
        erm_errors.append(np.mean(e_errs))

    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, moment_errors, marker='o', label='Moment Estimator')
    plt.plot(sample_sizes, erm_errors, marker='s', label='Net-ERM')
    plt.xlabel('Sample Size (n)')
    plt.ylabel(r'Orbit Error $d_{\mathcal{G}}$')
    plt.title('Folded Normal: Estimator Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/folded_normal_py.pdf')
    plt.close()

# --- GMM Experiment ---
def generate_gmm(w, mu, sigma, n):
    components = np.random.choice(len(w), size=n, p=w)
    data = np.zeros(n)
    for i in range(len(w)):
        idx = (components == i)
        data[idx] = np.random.normal(mu[i], sigma[i], size=np.sum(idx))
    return data

def orbit_dist_gmm(w1, mu1, sig1, w2, mu2, sig2):
    # k=2
    # D(theta1, theta2) = min_{pi} d(theta1, pi * theta2)
    # d(theta, theta') = ||w-w'||_1 + max_j ||mu_j - mu'_j||_2 + max_j ||sig_j - sig'_j||_F

    best_dist = np.inf
    for perm in itertools.permutations([0, 1]):
        w2_perm = np.array([w2[i] for i in perm])
        mu2_perm = np.array([mu2[i] for i in perm])
        sig2_perm = np.array([sig2[i] for i in perm])

        dist = np.sum(np.abs(w1 - w2_perm)) + \
               np.max(np.abs(mu1 - mu2_perm)) + \
               np.max(np.abs(sig1 - sig2_perm))
        if dist < best_dist:
            best_dist = dist
    return best_dist

def run_gmm_experiments():
    true_w = np.array([0.4, 0.6])
    true_mu = np.array([-2.0, 2.0])
    true_sig = np.array([1.0, 1.0])

    sample_sizes = [100, 500, 1000, 2000, 5000, 10000]
    em_errors = []

    for n in sample_sizes:
        e_errs = []
        for _ in range(10):
            data = generate_gmm(true_w, true_mu, true_sig, n)
            data_reshaped = data.reshape(-1, 1)

            # Standard EM via scikit-learn
            gmm = GaussianMixture(n_components=2, covariance_type='diag', max_iter=200, random_state=None)
            gmm.fit(data_reshaped)

            w_hat = gmm.weights_
            mu_hat = gmm.means_.flatten()
            sig_hat = np.sqrt(gmm.covariances_.flatten())

            e_errs.append(orbit_dist_gmm(w_hat, mu_hat, sig_hat, true_w, true_mu, true_sig))

        em_errors.append(np.mean(e_errs))

    plt.figure(figsize=(6, 4))
    plt.plot(sample_sizes, em_errors, marker='^', color='green', label='EM Algorithm')
    plt.xlabel('Sample Size (n)')
    plt.ylabel(r'Orbit Error $d_{\mathcal{G}}$')
    plt.title('Gaussian Mixture (k=2): EM Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('experiments/gmm_py.pdf')
    plt.close()

if __name__ == '__main__':
    run_folded_normal_experiments()
    run_gmm_experiments()
