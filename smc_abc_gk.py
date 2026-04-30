"""
============================================================================
    SMC-ABC adapted to the simple hierarchical G&K model
    + Correlated Pseudo-Marginal on G&K noise
============================================================================

Model (Clarté et al. 2020 paper, Section 4, pp. 7-8):

    alpha     ~ U[-10, 10]
    mu_i      ~ N(alpha, 1)        i = 1..n
    x_i       ~ gk(mu_i, B, g, k)  observations of length K

Adaptation relative to `smc_abc.py`:

  * Summary statistic: OCTILES per group, MAE distance summed over the
    n groups (Section 4 p. 7:
        d(x, x*) = sum_i sum_{j=0..8} |q(x_i, j/8) - q(x*_i, j/8)|).
    We sample theta = (alpha, mu_1, ..., mu_n) JOINTLY in dimension
    n+1 -- this is exactly the pitfall highlighted in the paper (Figure 4 p. 10):
    in high dimensions, the global distance causes SMC-ABC to hit the
    wall of the curse of dimensionality.

  * Correlated Pseudo-Marginal (Deligiannidis, Doucet, Pitt 2018) maintained:
    we store auxiliary variables Z ~ N(0,1) that feed the
    G&K simulator via z = Phi^{-1}(U). Since z itself is ~ N(0,1), we
    store it directly -- no need to transform U <-> z. The
    AR(1) coupling is therefore:

        Z* = ρ · Z  +  √(1-ρ²) · ξ,    ξ ~ N(0,I).

  * Robust Cholesky: identical to `smc_abc.py`.

Main algorithm: Algorithm 5 from the paper's Supplement (p. 24), in
its adaptive version (Del Moral et al. 2012) with an adaptive Gaussian
kernel (Toni et al. 2008, Section 9 p. 24).
============================================================================
"""

from __future__ import annotations

import time

import numpy as np

from abc_gibbs import ABCResult
from gk_model import simulate_gk, octiles


# ===========================================================================
#                          JOINT PRIOR  pi(theta)
# ===========================================================================
# The Simple Hierarchical G&K model (paper Section 4, eq. (3) p. 7) formally
# FIXES Var(mu_i | alpha) = 1:
#                   mu_i ~ N(alpha, 1).
# We therefore remove any degree of freedom for sigma_mu to prevent a
# user from modifying this parameter, which is locked by the model
# specification. The constant GK_SIGMA_MU = 1.0 is hardcoded throughout.
GK_SIGMA_MU = 1.0


def sample_prior(N, n, alpha_low, alpha_high):
    alpha = np.random.uniform(alpha_low, alpha_high, size=N)
    mu = np.random.normal(alpha[:, None], GK_SIGMA_MU, size=(N, n))
    return np.column_stack([alpha, mu])


def log_prior(theta, alpha_low, alpha_high):
    a = theta[..., 0]
    mu = theta[..., 1:]
    inside = (a >= alpha_low) & (a <= alpha_high)
    # Note: sigma_mu = 1 (paper eq. (3) p. 7) -> denominator = 1.
    log_p = -0.5 * np.sum((mu - a[..., None]) ** 2, axis=-1)
    return np.where(inside, log_p, -np.inf)


# ===========================================================================
#       DETERMINISTIC SIMULATOR in (theta, Z) for Correlated PM
# ===========================================================================
def simulate_distances_gk(thetas, Z, B, g, k, q_obs):
    """
    For each particle theta_i and each pseudo-dataset m, deterministically
    simulates x = gk(mu_i, B, g, k ; z = Z[i,m]) and computes the
    MAE octile distance summed over the n groups (paper p. 7).

    thetas : (N, n+1)
    Z      : (N, M, n, K)   auxiliary variables N(0,1) -- G&K randomness
    q_obs  : (n, 9)          precomputed octiles

    Returns dist (N, M).
    """
    mu = thetas[:, 1:]                                    # (N, n)
    # Deterministic G&K in (mu, z)
    x = simulate_gk(mu[:, None, :, None], B, g, k, z=Z)   # (N, M, n, K)
    q_sim = octiles(x, axis=-1)                           # (N, M, n, 9)
    # Sum over 9 octiles then over n groups
    return np.sum(np.abs(q_sim - q_obs), axis=(-1, -2))   # (N, M)


# ===========================================================================
#               ADAPTIVE SELECTION OF THE THRESHOLD eps_j
# ===========================================================================
def _ess(w):
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    return 1.0 / np.sum(p ** 2)


def find_next_eps(dist_prev, eps_prev, w_prev, target_ess, tol=1e-10):
    """Bisection: smallest ε such that ESS({w_j(ε)}) >= target_ess
    (Algorithm 5 p. 24)."""
    counts_prev = (dist_prev < eps_prev).sum(axis=1).astype(float)

    def ess_at(eps):
        counts = (dist_prev < eps).sum(axis=1).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(counts_prev > 0, counts / counts_prev, 0.0)
        w_new = w_prev * ratio
        return _ess(w_new), w_new

    e_hi, w_hi = ess_at(eps_prev)
    if e_hi < target_ess:
        return eps_prev, w_hi
    lo, hi = 0.0, eps_prev
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        e_mid, _ = ess_at(mid)
        if e_mid >= target_ess:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    _, w_final = ess_at(hi)
    return hi, w_final


# ===========================================================================
#                       SYSTEMATIC RESAMPLING
# ===========================================================================
def systematic_resample(w):
    N = w.size
    p = w / w.sum()
    positions = (np.arange(N) + np.random.uniform()) / N
    cum = np.cumsum(p)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions)


# ===========================================================================
#               ROBUST CHOLESKY (dynamic jitter + fallback)
# ===========================================================================
def safe_cholesky(cov, base_jitter=1e-10, n_tries=20):
    d = cov.shape[0]
    scale = max(float(np.diag(cov).max()), 1e-12)
    jitter = base_jitter * scale
    eye = np.eye(d)
    for _ in range(n_tries):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-12)))


# ===========================================================================
#            MOVE STEP: MH with Correlated Pseudo-Marginal on Z
# ===========================================================================
def move_step(thetas, Z, dist_M, eps_j, kernel_chol, rho,
              alpha_low, alpha_high, B, g, k, q_obs):
    """
    Particle-wise MH targeting pi_{eps_j}(theta) ∝ pi(theta) ·
    P{ d(s(x),s(x*)) < eps_j | theta }, with AR(1) coupling on Z:

        theta* = theta + L · Z_theta
        Z*     = rho · Z + sqrt(1-rho^2) · xi,    xi ~ N(0,I)

    The G&K simulator is then deterministic in (theta*, Z*), which
    cancels the dominant variance of the binary MH ratio and stabilizes
    the chain (Deligiannidis, Doucet, Pitt 2018).

    The MH ratio uses STORED n_pass_cur (Andrieu & Roberts 2009) -- it
    is NOT re-simulated.
    """
    N, dim = thetas.shape

    # Proposal theta*
    Zt = np.random.normal(size=(N, dim))
    thetas_prop = thetas + Zt @ kernel_chol.T

    # AR(1) coupled Z* proposal
    xi = np.random.normal(size=Z.shape)
    Z_prop = rho * Z + np.sqrt(1.0 - rho ** 2) * xi

    # Deterministic G&K simulation
    dist_prop = simulate_distances_gk(thetas_prop, Z_prop, B, g, k, q_obs)

    n_pass_cur = (dist_M < eps_j).sum(axis=1).astype(float)
    n_pass_prop = (dist_prop < eps_j).sum(axis=1).astype(float)

    lp_cur = log_prior(thetas, alpha_low, alpha_high)
    lp_prop = log_prior(thetas_prop, alpha_low, alpha_high)

    with np.errstate(divide="ignore"):
        log_ratio = (lp_prop - lp_cur
                     + np.log(np.maximum(n_pass_prop, 1e-300))
                     - np.log(np.maximum(n_pass_cur, 1e-300)))
    log_ratio = np.where(np.isfinite(lp_prop) & (n_pass_prop > 0),
                         log_ratio, -np.inf)
    accept = np.log(np.random.uniform(size=N)) < log_ratio

    thetas_new = np.where(accept[:, None], thetas_prop, thetas)
    Z_new = np.where(accept[:, None, None, None], Z_prop, Z)
    dist_new = np.where(accept[:, None], dist_prop, dist_M)

    n_sims = N * Z.shape[1] * Z.shape[2] * Z.shape[3]
    return thetas_new, Z_new, dist_new, int(accept.sum()), n_sims


# ===========================================================================
#                                MAIN LOOP
# ===========================================================================
def smc_abc_gk(
    x_obs,
    B=1.0, g=0.2, k=0.5,
    N=500,
    T=20,
    M=1,
    alpha_ess=0.95,
    N_min_frac=0.5,
    n_move_steps=2,
    rho=0.99,
    alpha_low=-10.0,
    alpha_high=10.0,
    seed=None,
) -> ABCResult:
    """
    Adaptive SMC-ABC (Supplement Algorithm 5 p. 24) for hierarchical G&K,
    with Correlated Pseudo-Marginal (Deligiannidis et al. 2018).

    Note: sigma_mu = 1 is hardcoded (paper eq. (3) p. 7) -- this
    parameter does NOT appear in the signature to prevent any
    accidental out-of-spec modification.
    """
    if seed is not None:
        np.random.seed(seed)

    n, K = x_obs.shape
    q_obs = octiles(x_obs, axis=1)              # (n, 9) computed once
    N_min = int(N_min_frac * N)

    t0 = time.perf_counter()

    thetas = sample_prior(N, n, alpha_low, alpha_high)
    Z = np.random.normal(size=(N, M, n, K))
    dist_M = simulate_distances_gk(thetas, Z, B, g, k, q_obs)
    eps = dist_M.max()
    w = np.ones(N)

    n_sims = N * (n + M * n * K)
    eps_history = [eps]
    accept_history = []

    for j in range(1, T + 1):
        target_ess = alpha_ess * _ess(w)
        eps_new, w_new = find_next_eps(dist_M, eps, w, target_ess)
        if w_new.sum() <= 0 or eps_new >= eps:
            break
        eps = eps_new
        w = w_new

        if _ess(w) < N_min:
            idx = systematic_resample(w)
            thetas = thetas[idx]
            Z = Z[idx]
            dist_M = dist_M[idx]
            w = np.ones(N)

        wp = w / w.sum()
        mean_w = (wp[:, None] * thetas).sum(axis=0)
        diff = thetas - mean_w
        cov_w = (wp[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(0)
        kernel_chol = safe_cholesky(2.0 * cov_w)

        n_acc_total = 0
        for _ in range(n_move_steps):
            thetas, Z, dist_M, n_acc, ns = move_step(
                thetas, Z, dist_M, eps, kernel_chol, rho,
                alpha_low, alpha_high, B, g, k, q_obs,
            )
            n_acc_total += n_acc
            n_sims += ns

        eps_history.append(eps)
        accept_history.append(n_acc_total / (N * n_move_steps))

    idx = systematic_resample(w)
    thetas_final = thetas[idx]
    cpu_time = time.perf_counter() - t0

    return ABCResult(
        alpha_chain=thetas_final[:, 0],
        mu_chain=thetas_final[:, 1:],
        accepted_dists_mu=np.asarray(accept_history),
        accepted_dists_alpha=np.asarray(eps_history),
        cpu_time=cpu_time,
        n_model_sims=n_sims,
        config=dict(
            algorithm="SMC-ABC-GK",
            N=N, T=T, M=M, alpha_ess=alpha_ess, N_min_frac=N_min_frac,
            n_move_steps=n_move_steps, rho=rho, n=n, K=K,
            sigma_mu=GK_SIGMA_MU, B=B, g=g, k=k,    # sigma_mu hardcoded eq.(3)
            alpha_low=alpha_low, alpha_high=alpha_high, seed=seed,
            n_steps_done=len(eps_history) - 1,
            final_eps=float(eps_history[-1]),
            mean_accept_rate=float(np.mean(accept_history))
            if accept_history else float("nan"),
        ),
    )