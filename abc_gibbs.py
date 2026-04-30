"""
============================================================================
   ABC-Gibbs : Component-wise Approximate Bayesian Computation
                via Gibbs-like steps
============================================================================

Python implementation of **Algorithm 4** from the paper:

    Clarté G., Robert C.P., Ryder R.J., Stoehr J. (2020).
    "Component-wise Approximate Bayesian Computation via Gibbs-like steps"
    Biometrika, arXiv:1905.13599v5.

Model: Hierarchical Normal-Normal (paper Section 3.2, equation (2), p.6):

        mu_j   ~ N(alpha, ς^2)         i.i.d.  j = 1..n
        x_{jk} ~ N(mu_j, sigma^2)      i.i.d.  k = 1..K
        alpha  ~ U[-4, 4]              (hyper-prior, paper p.6)

For comparison tools (metrics, exact posterior, replicates), see the 
`comparison.py` module. This file contains only the ABC-Gibbs algorithm 
itself and the shared result structure `ABCResult`.
============================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


# ===========================================================================
#                STRUCTURE DE RÉSULTAT (partagée avec SMC-ABC)
# ===========================================================================
@dataclass
class ABCResult:
    """Conteneur des sorties d'un run ABC + métriques de coût."""
    alpha_chain: np.ndarray             # (T,)    échantillon postérieur
    mu_chain: np.ndarray                # (T, n)
    accepted_dists_mu: np.ndarray       # diagnostics par itération
    accepted_dists_alpha: np.ndarray
    cpu_time: float                     # secondes (time.perf_counter)
    n_model_sims: int                   # cf. Ntot du papier (p.6)
    config: dict = field(default_factory=dict)


# ===========================================================================
#                  ÉTAPE 1 :  mu_j  |  alpha   (vectorisé, vrai ABC)
# ===========================================================================
def gibbs_step_mu(x_obs, alpha, sigma_mu, sigma_obs, N_mu):
    """
    Vectorized update of the vector mu = (mu_1,...,mu_n) | alpha, x*.

    Reference: Algorithm 4 p.6, inner loop `for j = 1..n`.

    Implementation Choices
    ----------------------
    * Strictly *likelihood-free*: for each candidate mu^c, we actually
      simulate K pseudo-observations x^c ~ N(mu^c, sigma^2)^{⊗K} and then
      compute the summary statistic s_mu(x^c) = empirical mean. The
      analytical distribution N(mu, sigma^2/K) of a Gaussian sample mean
      is NOT used (it does not generalize to models like MA_2 where the
      summary statistic lacks a closed form).
    * Full vectorization: no Python loops over j or over the ABC
      candidates; all draws are performed on contiguous arrays.
    """
    n, K = x_obs.shape
    s_obs = x_obs.mean(axis=1)                                       # (n,)

    # mu^c_{j,c} ~ N(alpha, ς^2)         (eq. (2) p.6)               (n, N_mu)
    mu_cand = np.random.normal(alpha, sigma_mu, size=(n, N_mu))

    # x^c_{j,c,k} ~ N(mu^c_{j,c}, sigma^2)   (likelihood-free)
    x_pseudo = np.random.normal(
        loc=mu_cand[:, :, None], scale=sigma_obs, size=(n, N_mu, K)
    )
    s_sim = x_pseudo.mean(axis=2)                                    # (n, N_mu)

    dist = np.abs(s_sim - s_obs[:, None])                            # (n, N_mu)

    # Sélection ABC : on garde le candidat de plus petite distance par
    # composante (papier Section 2.3 fin p.4 : eps_j = quantile empirique
    # le plus bas).
    idx = np.argmin(dist, axis=1)
    rows = np.arange(n)
    mu_new = mu_cand[rows, idx]
    min_dists = dist[rows, idx]

    # Coût en simulations du modèle (papier p.6 :
    # "each iteration costs N_alpha*n + N_mu*n*K").
    n_sims = n * N_mu * (1 + K)
    return mu_new, min_dists, n_sims


# ===========================================================================
#                  ÉTAPE 2 :  alpha  |  mu      (vectorisé, vrai ABC)
# ===========================================================================
def gibbs_step_alpha(mu, sigma_mu, N_alpha, alpha_low=-4.0, alpha_high=4.0):
    """
    Vectorized update of alpha | mu (Algorithm 4, p.6).

    Similar to `gibbs_step_mu`, we actually simulate the n pseudo-mu's,
    and then compute the empirical summary statistic. No loops are used;
    everything is handled via a single vectorized call to the generator.
    """
    n = mu.size
    s_obs = mu.mean()                       # s_alpha sufficient (papier p.6)

    # alpha^c ~ U[-4, 4]
    alpha_cand = np.random.uniform(alpha_low, alpha_high, size=N_alpha)

    # Pseudo-mu : mu^c_{c,j} ~ N(alpha^c_c, ς^2) puis statistique = moyenne
    mu_pseudo = np.random.normal(
        loc=alpha_cand[:, None], scale=sigma_mu, size=(N_alpha, n)
    )
    s_sim = mu_pseudo.mean(axis=1)
    dist = np.abs(s_sim - s_obs)
    idx = int(np.argmin(dist))

    n_sims = N_alpha * (1 + n)
    return alpha_cand[idx], dist[idx], n_sims


# ===========================================================================
#                       BOUCLE PRINCIPALE  ABC-Gibbs
# ===========================================================================
def abc_gibbs(
    x_obs,
    sigma_mu,
    sigma_obs,
    N_iter=1000,
    N_mu=30,
    N_alpha=30,
    alpha_init=0.0,
    mu_init=None,
    alpha_low=-4.0,
    alpha_high=4.0,
    burn_in=5,
    seed=None,
) -> ABCResult:
    """
    Algorithm 4 from the paper (p.6).

    Parameters
    ----------
    x_obs         : (n, K) observed data
    sigma_mu      : ς in mu_j ~ N(alpha, ς^2)
    sigma_obs     : sigma in x_{jk} ~ N(mu_j, sigma^2)
    N_iter        : number of Gibbs iterations (paper Figure 1: N=1000)
    N_mu, N_alpha : sizes of the reference tables (paper p.7: ≈ 30)
    burn_in       : number of discarded iterations (paper p.7: "first 5 points removed")

    Returns
    -------
    ABCResult : alpha_chain, mu_chain, retained distances, CPU time, n_sims.
    """
    if seed is not None:
        np.random.seed(seed)
    n, K = x_obs.shape

    alpha_chain = np.empty(N_iter + 1)
    mu_chain = np.empty((N_iter + 1, n))
    dists_mu = np.empty((N_iter, n))
    dists_alpha = np.empty(N_iter)

    alpha_chain[0] = alpha_init
    mu_chain[0] = (
        mu_init if mu_init is not None
        else np.random.normal(alpha_init, sigma_mu, size=n)
    )

    n_sims = 0
    t0 = time.perf_counter()

    # papier, Algorithm 4 : "for i = 1..N do"
    for i in range(1, N_iter + 1):
        mu_new, dmu, c1 = gibbs_step_mu(
            x_obs, alpha_chain[i - 1], sigma_mu, sigma_obs, N_mu
        )
        mu_chain[i] = mu_new
        dists_mu[i - 1] = dmu

        a_new, da, c2 = gibbs_step_alpha(
            mu_chain[i], sigma_mu, N_alpha, alpha_low, alpha_high
        )
        alpha_chain[i] = a_new
        dists_alpha[i - 1] = da

        n_sims += c1 + c2

    cpu_time = time.perf_counter() - t0

    return ABCResult(
        alpha_chain=alpha_chain[burn_in + 1:],
        mu_chain=mu_chain[burn_in + 1:],
        accepted_dists_mu=dists_mu,
        accepted_dists_alpha=dists_alpha,
        cpu_time=cpu_time,
        n_model_sims=n_sims,
        config=dict(
            algorithm="ABC-Gibbs",
            N_iter=N_iter, N_mu=N_mu, N_alpha=N_alpha, n=n, K=K,
            sigma_mu=sigma_mu, sigma_obs=sigma_obs,
            burn_in=burn_in, seed=seed,
        ),
    )
