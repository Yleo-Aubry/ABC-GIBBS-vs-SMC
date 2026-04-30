"""
============================================================================
   Comparison of ABC-Gibbs vs SMC-ABC on the hierarchical G&K model
============================================================================

Reference: Clarté G., Robert C.P., Ryder R.J., Stoehr J. (2020).
            "Component-wise Approximate Bayesian Computation via Gibbs-like
             steps."  Biometrika.  Section 4 (p.7-9), Figures 4-6.

The TRUE posterior is unknown here (G&K = intractable likelihood),
so we CANNOT compute a bias or a Wasserstein distance against an
analytical reference. We replace it with the standard metric in the
ABC domain:

  * **posterior predictive distance**: for each sample (alpha_s, mu_s)
    from the chain, we simulate fresh G&K data and compute the octile
    distance to the true observations. A good pseudo-posterior produces,
    on average, data close to x* (paper Section 10.1 p.5 of the supplement:
        "we simulate new synthetic data from each parameter set in the
         output, and compute the distance ... If ABC-Gibbs produces a
         smaller value than the ABC sampler ..., this is an indicator
         of a better fit of the ABC-Gibbs distribution with the true
         posterior.")

Retained metrics:
  * `cpu_time`     -- wall-clock time
  * `n_model_sims` -- simulation cost (cf. paper p.6)
  * `mc_error`     -- intra-chain Monte Carlo error (batch means)
  * `run_replicates` -- inter-run variability (empirical MC-error)

Run the comparison via: `python comparison_gk.py`.
============================================================================
"""

from __future__ import annotations

import numpy as np

from abc_gibbs import ABCResult
from gk_model import simulate_gk, octiles
from abc_gibbs_gk import abc_gibbs_gk
from smc_abc_gk import smc_abc_gk


# ===========================================================================
#                  DISTANCE PRÉDICTIVE POSTÉRIEURE
# ===========================================================================
def posterior_predictive_distance(
    result: ABCResult, x_obs, B, g, k,
    n_pred_samples=200, seed=0,
):
    """
    For S samples (alpha_s, mu_s) drawn without replacement from the posterior
    chain, simulates fresh G&K pseudo-data and computes the MAE octile
    distance summed over the n groups (statistic from the paper, Section 4 p.7).

    Returns the mean, standard deviation, and median over the S samples.
    A SMALLER predictive distance indicates a pseudo-posterior that
    better reproduces the observed data, and thus a better fit.
    """
    rng = np.random.default_rng(seed)
    n, K = x_obs.shape
    q_obs = octiles(x_obs, axis=1)                  # (n, 9)

    T = result.alpha_chain.shape[0]
    if T > n_pred_samples:
        sub = rng.choice(T, n_pred_samples, replace=False)
    else:
        sub = np.arange(T)

    mu_post = result.mu_chain[sub]                  # (S, n)
    S = mu_post.shape[0]

    # Pseudo-data G&K via aléas frais Z ~ N(0,1)    (S, n, K)
    z = rng.standard_normal(size=(S, n, K))
    x_pred = simulate_gk(mu_post[:, :, None], B, g, k, z=z)

    q_pred = octiles(x_pred, axis=-1)                            # (S, n, 9)
    dist = np.sum(np.abs(q_pred - q_obs), axis=(-1, -2))         # (S,)

    return dict(
        mean_pred_dist=float(dist.mean()),
        std_pred_dist=float(dist.std(ddof=1)),
        median_pred_dist=float(np.median(dist)),
        q05_pred_dist=float(np.quantile(dist, 0.05)),
        q95_pred_dist=float(np.quantile(dist, 0.95)),
    )


# ===========================================================================
#                  MONTE CARLO ERROR  (batch means)
# ===========================================================================
def batch_means_se(chain, n_batches=20):
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
    se_mu = batch_means_se(result.mu_chain)
    return dict(
        se_alpha=batch_means_se(result.alpha_chain),
        se_mu=se_mu,
        mean_se_mu=float(np.mean(se_mu)),
    )


# ===========================================================================
#               MULTI-RUN : MC-error empirique sur réplicats
# ===========================================================================
def run_replicates(run_fn, n_replicates=5, base_seed=0, **kwargs):
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
#                         AFFICHAGE
# ===========================================================================
def summarize_run(name, res, x_obs, B, g, k):
    pp = posterior_predictive_distance(res, x_obs, B, g, k)
    mce = mc_error(res)
    print(f"--- {name} ---")
    print(f"  CPU time            : {res.cpu_time:7.3f} s")
    print(f"  Coût (sims modèle)  : {res.n_model_sims:>14,}")
    print(f"  Pred dist (mean±sd) : {pp['mean_pred_dist']:.3f} "
          f"± {pp['std_pred_dist']:.3f}")
    print(f"  Pred dist (median)  : {pp['median_pred_dist']:.3f}")
    print(f"  Pred dist (5%-95%)  : [{pp['q05_pred_dist']:.3f}, "
          f"{pp['q95_pred_dist']:.3f}]")
    print(f"  MC-SE alpha (BM)    : {mce['se_alpha']:.4f}")
    print(f"  MC-SE moy. mu (BM)  : {mce['mean_se_mu']:.4f}")
    if "final_eps" in res.config:
        print(f"  eps final           : {res.config['final_eps']:.3f}"
              f"  ({res.config['n_steps_done']} étapes SMC)")
    print()


def summarize_replicates(name, rep, x_obs, B, g, k):
    print(f"--- {name} ---")
    print(f"  CPU time (mean ± sd) : {rep['cpu_time_mean']:.3f}"
          f" ± {rep['cpu_time_std']:.3f} s")
    print(f"  Coût (sims modèle)   : {rep['n_model_sims']:,}")
    print(f"  MC-SE empirique alpha: {rep['mc_se_alpha']:.4f}")
    print(f"  MC-SE empirique mu   : {rep['mean_mc_se_mu']:.4f}")
    # Pred dist agrégée sur tous les runs
    pps = [posterior_predictive_distance(r, x_obs, B, g, k)["mean_pred_dist"]
           for r in rep["results"]]
    print(f"  Pred dist (across runs): {np.mean(pps):.3f}"
          f" ± {np.std(pps, ddof=1):.3f}")
    print()


# ===========================================================================
#                                MAIN
# ===========================================================================
if __name__ == "__main__":
    # ----- Configuration Section 4 du papier (p.7) ------------------------
    # Simple hierarchical G & K : n = 50, alpha ~ U[-10, 10], mu_i ~ N(α, 1)
    # Paramètres G&K connus : B=1, g=0.2, k=0.5 (réglage standard).
    n = 50
    K = 100
    B, g, k = 1.0, 0.2, 0.5
    sigma_mu = 1.0
    alpha_true = 3.0

    rng = np.random.default_rng(42)
    mu_true = rng.normal(alpha_true, sigma_mu, size=n)
    z_obs = rng.standard_normal(size=(n, K))
    x_obs = simulate_gk(mu_true[:, None], B, g, k, z=z_obs)

    print("=" * 70)
    print(" RUN UNIQUE -- ABC-Gibbs vs SMC-ABC  (Simple Hierarchical G&K)")
    print(f" n={n}, K={K}, B={B}, g={g}, k={k}, alpha_true={alpha_true}")
    print("=" * 70)

    # Budgets choisis pour rester dans le même ordre de coût en sims modèle
    res_g = abc_gibbs_gk(
        x_obs, sigma_mu=sigma_mu, B=B, g=g, k=k,
        N_iter=300, N_mu=30, N_alpha=30, burn_in=10, seed=0,
    )
    res_s = smc_abc_gk(
        x_obs, B=B, g=g, k=k,                       # sigma_mu figé à 1 (eq.3)
        N=500, T=20, M=1, alpha_ess=0.95,
        N_min_frac=0.5, n_move_steps=2, rho=0.99, seed=0,
    )

    summarize_run("ABC-Gibbs", res_g, x_obs, B, g, k)
    summarize_run("SMC-ABC",   res_s, x_obs, B, g, k)

    print("=" * 70)
    print(" RÉPLICATS INDÉPENDANTS  --  variabilité inter-runs")
    print("=" * 70)

    rep_g = run_replicates(
        abc_gibbs_gk, n_replicates=3, base_seed=100,
        x_obs=x_obs, sigma_mu=sigma_mu, B=B, g=g, k=k,
        N_iter=300, N_mu=30, N_alpha=30, burn_in=10,
    )
    rep_s = run_replicates(
        smc_abc_gk, n_replicates=3, base_seed=100,
        x_obs=x_obs, B=B, g=g, k=k,                 # sigma_mu figé à 1 (eq.3)
        N=500, T=20, M=1, alpha_ess=0.95,
        N_min_frac=0.5, n_move_steps=2, rho=0.99,
    )
    summarize_replicates("ABC-Gibbs", rep_g, x_obs, B, g, k)
    summarize_replicates("SMC-ABC",   rep_s, x_obs, B, g, k)
