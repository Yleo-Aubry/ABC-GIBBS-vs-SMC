"""
============================================================================
   Comparison of ABC-Gibbs vs SMC-ABC
   on the hierarchical Normal-Normal model (Section 3.2 of the paper)
============================================================================

Reference:
    Clarté G., Robert C.P., Ryder R.J., Stoehr J. (2020).
    "Component-wise Approximate Bayesian Computation via Gibbs-like steps"
    Biometrika, arXiv:1905.13599v5.

This module provides:

  * `true_posterior_normal_normal` : the TRUE posterior (closed form,
        analytical integration of mu) for the model (eq. (2) p.6).
        Serves as a baseline to compute the inferential error.
  * `inferential_error` : bias, variance ratio, Wasserstein-1 distance
        between the pseudo-posterior and the true posterior.
        Link to the paper: Section 6 / Theorem 3 (||νε - ν0||).
  * `mc_error` : intra-chain Monte Carlo error via "batch means".
  * `run_replicates` : empirical inter-run MC-error (variability of
        the estimator across independent replicates -- the most
        honest metric to compare stochastic algorithms).

Running `python comparison.py` launches the ABC-Gibbs vs SMC-ABC comparison
using the configuration from Figure 1 of the paper (sigma=1, K=10, n=20).
============================================================================
"""

from __future__ import annotations

import numpy as np

from abc_gibbs import ABCResult, abc_gibbs
from smc_abc import smc_abc


# ===========================================================================
#               POSTÉRIEURE EXACTE (forme close pour ce modèle)
# ===========================================================================
def true_posterior_normal_normal(x_obs, sigma_mu, sigma_obs,
                                 alpha_low=-4.0, alpha_high=4.0,
                                 n_grid=4001, n_samples=20000, seed=0):
    """
    Normal-Normal model from eq. (2) p.6 of the paper. By integrating out mu
    analytically:

        x̄_j | alpha ~ N(alpha, ς^2 + sigma^2/K)

    hence the posterior of alpha (up to a normalizing constant):

        p(alpha | x*) ∝ 1_{[-4,4]}(alpha) · prod_j phi(x̄_j; alpha, τ²),
        τ² = ς² + σ²/K.

    Computation on a fine grid + sampling via CDF inversion.
    The posterior of mu_j | alpha, x* is a closed-form Gaussian.

    Returns: alpha_grid, alpha_pdf, alpha_mean, alpha_var, alpha_samples,
             mu_mean, mu_var, mu_samples.
    """
    n, K = x_obs.shape
    x_bar = x_obs.mean(axis=1)
    tau2 = sigma_mu ** 2 + sigma_obs ** 2 / K

    a_grid = np.linspace(alpha_low, alpha_high, n_grid)
    diff = x_bar[None, :] - a_grid[:, None]
    log_p = -0.5 * np.sum(diff ** 2, axis=1) / tau2
    log_p -= log_p.max()
    pdf = np.exp(log_p)
    pdf /= np.trapezoid(pdf, a_grid)

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                           * np.diff(a_grid))])
    cdf /= cdf[-1]

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n_samples)
    alpha_samples = np.interp(u, cdf, a_grid)

    a_mean = np.trapezoid(a_grid * pdf, a_grid)
    a_var = np.trapezoid((a_grid - a_mean) ** 2 * pdf, a_grid)

    # mu_j | alpha, x* ~ N(m_j(alpha), s²),
    # s² = (1/ς² + K/σ²)^-1, m_j(alpha) = s² (alpha/ς² + Σ_k x_jk/σ²)
    s2 = 1.0 / (1.0 / sigma_mu ** 2 + K / sigma_obs ** 2)
    mu_samples = np.empty((n_samples, n))
    for j in range(n):
        m_j = s2 * (alpha_samples / sigma_mu ** 2
                    + x_obs[j].sum() / sigma_obs ** 2)
        mu_samples[:, j] = rng.normal(m_j, np.sqrt(s2))

    return dict(
        alpha_grid=a_grid, alpha_pdf=pdf,
        alpha_mean=float(a_mean), alpha_var=float(a_var),
        alpha_samples=alpha_samples,
        mu_mean=mu_samples.mean(axis=0),
        mu_var=mu_samples.var(axis=0),
        mu_samples=mu_samples,
    )


# ===========================================================================
#                   MÉTRIQUE 1 : INFERENTIAL ERROR
# ===========================================================================
def wasserstein1_1d(a, b, n_q=1000):
    """
    Wasserstein-1 distance between two 1D samples.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    m = max(len(a), len(b), n_q)
    q = (np.arange(m) + 0.5) / m
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def inferential_error(result: ABCResult, truth: dict) -> dict:
    """
    Deviation between the ABC pseudo-posterior and the true posterior.

    Cf. Theorem 3 p.5: ||νε - ν0||_TV. We use numerically 
    accessible proxies: bias, variance ratio, and Wasserstein-1 distance.
    """
    a = result.alpha_chain
    bias_a = a.mean() - truth["alpha_mean"]
    var_ratio_a = a.var(ddof=1) / truth["alpha_var"]
    wass_a = wasserstein1_1d(a, truth["alpha_samples"])

    n = result.mu_chain.shape[1]
    bias_mu = result.mu_chain.mean(axis=0) - truth["mu_mean"]
    var_ratio_mu = result.mu_chain.var(axis=0, ddof=1) / truth["mu_var"]
    wass_mu = np.array([
        wasserstein1_1d(result.mu_chain[:, j], truth["mu_samples"][:, j])
        for j in range(n)
    ])

    return dict(
        bias_alpha=float(bias_a),
        var_ratio_alpha=float(var_ratio_a),
        wass_alpha=float(wass_a),
        bias_mu=bias_mu, var_ratio_mu=var_ratio_mu, wass_mu=wass_mu,
        mean_abs_bias_mu=float(np.mean(np.abs(bias_mu))),
        mean_wass_mu=float(np.mean(wass_mu)),
    )


# ===========================================================================
#                   MÉTRIQUE 2 : MONTE CARLO ERROR
# ===========================================================================
def batch_means_se(chain, n_batches=20):
    """
    Monte Carlo error of the mean estimator via "batch means":
    SE = sqrt( var(batch_means) / B ). Standard practice for MCMC.
    """
    chain = np.asarray(chain)
    T = chain.shape[0]
    b = T // n_batches
    if b == 0:
        return float("nan")
    trimmed = chain[: b * n_batches]
    if trimmed.ndim == 1:
        means = trimmed.reshape(n_batches, b).mean(axis=1)
        return float(np.sqrt(means.var(ddof=1) / n_batches))
    means = trimmed.reshape(n_batches, b, -1).mean(axis=1)
    return np.sqrt(means.var(axis=0, ddof=1) / n_batches)


def mc_error(result: ABCResult) -> dict:
    """MC intra-run error (batch means)."""
    se_mu = batch_means_se(result.mu_chain)
    return dict(
        se_alpha=batch_means_se(result.alpha_chain),
        se_mu=se_mu,
        mean_se_mu=float(np.mean(se_mu)),
    )


# ===========================================================================
#               MULTI-RUN : MC-error empirique sur réplicats
# ===========================================================================
def run_replicates(run_fn, n_replicates=10, base_seed=0, **kwargs):
    """
    Launches `n_replicates` independent runs of `run_fn(**kwargs, seed=...)`.
    """
    results = []
    means_a, means_mu = [], []
    for r in range(n_replicates):
        res = run_fn(seed=base_seed + r, **kwargs)
        results.append(res)
        means_a.append(res.alpha_chain.mean())
        means_mu.append(res.mu_chain.mean(axis=0))
    means_a = np.array(means_a)
    means_mu = np.array(means_mu)
    return dict(
        results=results,
        cpu_time_mean=float(np.mean([r.cpu_time for r in results])),
        cpu_time_std=float(np.std([r.cpu_time for r in results], ddof=1)),
        n_model_sims=results[0].n_model_sims,
        mc_se_alpha=float(means_a.std(ddof=1)),
        mc_se_mu=means_mu.std(axis=0, ddof=1),
        mean_mc_se_mu=float(np.mean(means_mu.std(axis=0, ddof=1))),
    )


# ===========================================================================
#                     TABLEAU RÉCAPITULATIF
# ===========================================================================
def summarize_run(name, res, truth):
    inf = inferential_error(res, truth)
    mce = mc_error(res)
    print(f"--- {name} ---")
    print(f"  CPU time          : {res.cpu_time:7.3f} s")
    print(f"  Coût (sims modèle): {res.n_model_sims:>12,}")
    print(f"  Bias  alpha       : {inf['bias_alpha']:+.4f}")
    print(f"  VarRatio alpha    : {inf['var_ratio_alpha']:.3f}  (1.0 = exact)")
    print(f"  Wass1 alpha       : {inf['wass_alpha']:.4f}")
    print(f"  |bias| moyen mu_j : {inf['mean_abs_bias_mu']:.4f}")
    print(f"  Wass1 moyen mu_j  : {inf['mean_wass_mu']:.4f}")
    print(f"  MC-SE alpha (BM)  : {mce['se_alpha']:.4f}")
    print(f"  MC-SE moy. mu (BM): {mce['mean_se_mu']:.4f}")
    if "final_eps" in res.config:
        print(f"  eps final         : {res.config['final_eps']:.4f}"
              f"  ({res.config['n_steps_done']} étapes SMC)")
    print()


def summarize_replicates(name, rep):
    print(f"--- {name} ---")
    print(f"  CPU time (mean ± sd) : {rep['cpu_time_mean']:.3f}"
          f" ± {rep['cpu_time_std']:.3f} s")
    print(f"  Coût (sims modèle)   : {rep['n_model_sims']:,}")
    print(f"  MC-SE empirique alpha: {rep['mc_se_alpha']:.4f}")
    print(f"  MC-SE empirique mu   : {rep['mean_mc_se_mu']:.4f}")
    print()


# ===========================================================================
#                                MAIN
# ===========================================================================
if __name__ == "__main__":
    # ----- Configuration Figure 1 du papier (p.8) -----------------------
    sigma_obs, sigma_mu = 1.0, 1.0
    n, K = 20, 10

    rng = np.random.default_rng(42)
    alpha_true = 2.5
    mu_true = rng.normal(alpha_true, sigma_mu, size=n)
    x_obs = rng.normal(mu_true[:, None], sigma_obs, size=(n, K))

    truth = true_posterior_normal_normal(x_obs, sigma_mu, sigma_obs)

    # =================================================================
    #                  RUN UNIQUE : ABC-Gibbs vs SMC-ABC
    # =================================================================
    print("=" * 65)
    print(" RUN UNIQUE  --  ABC-Gibbs vs SMC-ABC")
    print(" (config Section 3.2 du papier : sigma=1, sigma_mu=1, K=10, n=20)")
    print("=" * 65)

    res_g = abc_gibbs(x_obs, sigma_mu, sigma_obs,
                      N_iter=1000, N_mu=30, N_alpha=30, seed=0)
    res_s = smc_abc(x_obs, sigma_mu, sigma_obs,
                    N=1000, T=30, M=1, alpha_ess=0.95,
                    N_min_frac=0.5, n_move_steps=2, seed=0)

    summarize_run("ABC-Gibbs", res_g, truth)
    summarize_run("SMC-ABC",   res_s, truth)

    # =================================================================
    #         RÉPLICATS INDÉPENDANTS (MC-error empirique)
    # =================================================================
    print("=" * 65)
    print(" RÉPLICATS INDÉPENDANTS  --  MC-error empirique inter-runs")
    print("=" * 65)

    rep_g = run_replicates(
        abc_gibbs, n_replicates=5, base_seed=100,
        x_obs=x_obs, sigma_mu=sigma_mu, sigma_obs=sigma_obs,
        N_iter=1000, N_mu=30, N_alpha=30,
    )
    rep_s = run_replicates(
        smc_abc, n_replicates=5, base_seed=100,
        x_obs=x_obs, sigma_mu=sigma_mu, sigma_obs=sigma_obs,
        N=1000, T=30, M=1, alpha_ess=0.95,
        N_min_frac=0.5, n_move_steps=2,
    )
    summarize_replicates("ABC-Gibbs", rep_g)
    summarize_replicates("SMC-ABC",   rep_s)
