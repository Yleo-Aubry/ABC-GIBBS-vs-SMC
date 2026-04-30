"""
============================================================================
   ABC-Gibbs adapted to the simple hierarchical G & K model
============================================================================

Model (Clarté et al. 2020 paper, Section 4, p.7-8) :

    alpha     ~ U[-10, 10]
    mu_i      ~ N(alpha, 1)        i = 1..n
    x_i       ~ gk(mu_i, B, g, k)  observations of length K
                                   (B, g, k are known)

Adaptation compared to `abc_gibbs.py` :

  * Step  mu_i | alpha          -- the summary statistic is no longer the
                                   empirical mean (not sufficient for 
                                   G&K) but the **octiles** (Section 4
                                   p.7), using MAE distance.
  * Step  alpha | mu            -- UNCHANGED compared to the Normal-
                                   Normal case because the distribution 
                                   of mu_j | alpha remains Gaussian. 
                                   Sufficient statistic: empirical mean 
                                   of the mu_j's (paper Section 3.2 p.6).

Algorithm 4 from the paper remains valid here: this is exactly the 
illustration provided by the authors in Section 4. The G&K simulator is 
strictly likelihood-free and all operations remain vectorized via NumPy.
============================================================================"""

from __future__ import annotations

import time

import numpy as np

from abc_gibbs import ABCResult
from gk_model import simulate_gk, octiles, _OCTILE_PROBS


# ===========================================================================
#       ÉTAPE 1 :  mu_i | alpha   (vectorisée, simulateur G&K)
# ===========================================================================
def gibbs_step_mu_gk(x_obs, alpha, sigma_mu, B, g, k, N_mu, q_obs):
    """
    Vectorized update of mu = (mu_1,...,mu_n) | alpha, x* for the 
    hierarchical G&K model (Section 4 p.7).

    For each component i, we draw N_mu candidates mu^c_i ~ N(alpha, ς²),
    simulate K G&K pseudo-observations per candidate, compute the 
    octiles, and retain the candidate with the smallest MAE octile 
    distance to x*_i.

    Vectorization: no loops over i or the candidates.

    Parameters
    ----------
    x_obs    : (n, K)
    alpha    : float, current value
    sigma_mu : float, standard deviation of the conditional prior mu | alpha
    B,g,k    : G&K parameters (known, fixed)
    N_mu     : size of the reference table per component
    q_obs    : (n, 9) precomputed observed octiles (for computational efficiency)
    """
    n, K = x_obs.shape

    # Candidats mu^c_{i,c} ~ N(alpha, sigma_mu^2)              (n, N_mu)
    mu_cand = np.random.normal(alpha, sigma_mu, size=(n, N_mu))

    # Aléas auxiliaires gaussiens pour le simulateur G&K       (n, N_mu, K)
    z = np.random.normal(size=(n, N_mu, K))

    # Pseudo-observations G&K (mu broadcasté sur K)            (n, N_mu, K)
    x_pseudo = simulate_gk(mu_cand[:, :, None], B, g, k, z=z)

    # Octiles simulées                                          (n, N_mu, 9)
    q_sim = octiles(x_pseudo, axis=-1)

    # Distance MAE d'octiles : somme axe -1                     (n, N_mu)
    dist = np.sum(np.abs(q_sim - q_obs[:, None, :]), axis=-1)

    idx = np.argmin(dist, axis=1)
    rows = np.arange(n)
    mu_new = mu_cand[rows, idx]
    min_dists = dist[rows, idx]

    # Coût en simulations : tirages G&K + tirages prior
    n_sims = n * N_mu * K + n * N_mu
    return mu_new, min_dists, n_sims


# ===========================================================================
#       ÉTAPE 2 :  alpha | mu   (identique au cas Normal-Normal)
# ===========================================================================
def gibbs_step_alpha_gk(mu, sigma_mu, N_alpha, alpha_low=-10.0, alpha_high=10.0):
    """
    Update of alpha | mu. Since the conditional distribution of mu_j | alpha remains
    Gaussian in the hierarchical G&K model (Section 4 p.7), the sufficient
    statistic associated with alpha remains the empirical mean of the
    mu_j's (paper Section 3.2 p.6). No G&K simulation is required
    at this step.
    """
    n = mu.size
    s_obs = mu.mean()

    alpha_cand = np.random.uniform(alpha_low, alpha_high, size=N_alpha)
    mu_pseudo = np.random.normal(
        loc=alpha_cand[:, None], scale=sigma_mu, size=(N_alpha, n)
    )
    s_sim = mu_pseudo.mean(axis=1)
    dist = np.abs(s_sim - s_obs)
    idx = int(np.argmin(dist))

    n_sims = N_alpha * (1 + n)
    return alpha_cand[idx], dist[idx], n_sims


# ===========================================================================
#                       BOUCLE PRINCIPALE  ABC-Gibbs G&K
# ===========================================================================
def abc_gibbs_gk(
    x_obs,
    sigma_mu=1.0,
    B=1.0, g=0.2, k=0.5,
    N_iter=500,
    N_mu=30,
    N_alpha=30,
    alpha_init=0.0,
    mu_init=None,
    alpha_low=-10.0,
    alpha_high=10.0,
    burn_in=10,
    seed=None,
) -> ABCResult:
    """
    ABC-Gibbs for the simple hierarchical G&K model (Section 4 of the paper).

    Convention: sigma_mu = 1 (paper eq. (3) p.7).
    Prior bounds for alpha: [-10, 10] (paper p.7).
    """
    if seed is not None:
        np.random.seed(seed)

    n, K = x_obs.shape
    q_obs = octiles(x_obs, axis=1)             # (n, 9) calculé une seule fois

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

    for i in range(1, N_iter + 1):
        mu_new, dmu, c1 = gibbs_step_mu_gk(
            x_obs, alpha_chain[i - 1], sigma_mu, B, g, k, N_mu, q_obs
        )
        mu_chain[i] = mu_new
        dists_mu[i - 1] = dmu

        a_new, da, c2 = gibbs_step_alpha_gk(
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
            algorithm="ABC-Gibbs-GK",
            N_iter=N_iter, N_mu=N_mu, N_alpha=N_alpha, n=n, K=K,
            sigma_mu=sigma_mu, B=B, g=g, k=k,
            alpha_low=alpha_low, alpha_high=alpha_high,
            burn_in=burn_in, seed=seed,
        ),
    )
