"""
============================================================================
   SMC-ABC: Sequential Monte Carlo - Approximate Bayesian Computation
   + Correlated Pseudo-Marginal + Robust Cholesky
============================================================================

Implementation of **Algorithm 5** from the Supplement of the paper:

    Clarté, Robert, Ryder, Stoehr (2020).
    "Component-wise Approximate Bayesian Computation via Gibbs-like steps"
    Biometrika, arXiv:1905.13599v5 (Section 9, page 24 of the supplement).

Algorithm chosen by the authors as a sophisticated baseline against
ABC-Gibbs, an explicit fusion of two reference schemes:

  * Del Moral, Doucet, Jasra (2012)
       "An adaptive sequential Monte Carlo method for ABC."
       Statistics and Computing 22, 1009-1020.
       --> ADAPTIVE threshold $\epsilon_j$ via $ESS_j = \alpha \cdot ESS_{j-1}$ criterion,
           conditional resampling on $ESS < N_{min}$.

  * Toni, Welch, Strelkowa, Ipsen, Stumpf (2008)
       "ABC scheme for parameter inference and model selection in
        dynamical systems." J. R. Soc. Interface 6, 187-202.
       --> adaptive Gaussian kernel $K_j$ with covariance $2 \cdot Cov(particles)$
           (cf. Section 9 p.24 of the supplement).

----------------------------------------------------------------------------
Technical improvements over a naive implementation:

  [A] CORRELATED PSEUDO-MARGINAL (variance reduction of the MH ratio)
      ---------------------------------------------------------------
      Reference:
          Deligiannidis, Doucet, Pitt (2018)
          "The correlated pseudo-marginal method."
          J. R. Statist. Soc. B 80(5), 839-870.
          ALSO: Dahlin, Lindsten, Kronander, Schön (2015)
          "Accelerating pseudo-marginal Metropolis-Hastings by
           correlating auxiliary variables."

      With $M = 1$ per particle, the Monte Carlo estimator of the
      approximate likelihood $1\{ d < \epsilon_j \}$ is binary and has
      maximal variance -> "sticky chain" phenomenon (MCMC chain freezing).
      We correct this by coupling the pseudo-random numbers used
      to generate the current and proposed pseudo-datasets via
      an AR(1) scheme on the Gaussian auxiliary variables:

            $U^{(prop)} = \rho \cdot U^{(cur)} + \sqrt{1-\rho^2} \cdot Z, \quad Z \sim N(0,I)$

      with $\rho \in [0,1)$ close to 1. For $\theta^*$ close to $\theta$, the coupling
      ensures that both pseudo-datasets are correlated, making
      the transition between $n\_pass\_cur$ and $n\_pass\_prop$ continuous
      rather than jumping. The simulator then becomes entirely
      deterministic in $(\theta, U)$: $x = \mu + \sigma \cdot U$.

  [B] ROBUST CHOLESKY WITH ADAPTIVE REGULARIZATION
      -----------------------------------------------
      Reference:
          Higham (1988) "Computing a nearest symmetric positive
                         semidefinite matrix." Lin. Alg. Appl. 103.

      At the end of the SMC algorithm, the weighted covariance of the particles
      may lose its rank (weights concentrated on few particles).
      Instead of a static $1e-8 \cdot I$ jitter, we dynamically calibrate the
      jitter based on the scale (max-diag) of the matrix and increase by
      powers of 10 until Cholesky succeeds, with a
      diagonal fallback as a last resort. Gaussian sampling
      via $Z @ L.T$ (more stable than np.random.multivariate_normal).

----------------------------------------------------------------------------
Model (paper eq. (2) p.6): see abc_gibbs.py.

For comparison tools, see `comparison.py`.
============================================================================
"""

from __future__ import annotations

import time

import numpy as np

from abc_gibbs import ABCResult


# ===========================================================================
#                          JOINT PRIOR  pi(theta)
# ===========================================================================
def sample_prior(N, n, sigma_mu, alpha_low, alpha_high):
    """$\theta = (\alpha, \mu_1, ..., \mu_n)$; returns (N, n+1)."""
    alpha = np.random.uniform(alpha_low, alpha_high, size=N)
    mu = np.random.normal(alpha[:, None], sigma_mu, size=(N, n))
    return np.column_stack([alpha, mu])


def log_prior(theta, sigma_mu, alpha_low, alpha_high):
    """log $\pi(\theta)$; $-\infty$ outside the support of $\alpha$ (constant omitted)."""
    a = theta[..., 0]
    mu = theta[..., 1:]
    inside = (a >= alpha_low) & (a <= alpha_high)
    log_p = -0.5 * np.sum(((mu - a[..., None]) / sigma_mu) ** 2, axis=-1)
    return np.where(inside, log_p, -np.inf)


# ===========================================================================
#       DETERMINISTIC SIMULATOR: $x = \mu + \sigma \cdot U$ (noise coupling)
# ===========================================================================
def simulate_from_noise(thetas, U, sigma_obs, s_obs):
    """
    Deterministic simulator in $(\theta, U)$: for a fixed U, the pseudo-dataset is
    a smooth function of $\theta$. This is the key to the Correlated Pseudo-Marginal
    (Deligiannidis et al. 2018): exact noise coupling.

        $x_{i,m,j,k} = \mu_{i,j} + \sigma \cdot U_{i,m,j,k}$

    thetas : (N, n+1)
    U      : (N, M, n, K)   auxiliary variables N(0,1)
    Returns dist (N, M) as Euclidean distance on the vector of means.
    """
    mu = thetas[:, 1:]                                  # (N, n)
    x = mu[:, None, :, None] + sigma_obs * U            # (N, M, n, K)
    s_sim = x.mean(axis=3)                              # (N, M, n)
    return np.linalg.norm(s_sim - s_obs[None, None, :], axis=2)   # (N, M)


# ===========================================================================
#               ADAPTIVE THRESHOLD SELECTION eps_j (Del Moral 2012)
# ===========================================================================
def _ess(w):
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    return 1.0 / np.sum(p ** 2)


def find_next_eps(dist_prev, eps_prev, w_prev, target_ess, tol=1e-10):
    """
    Bisection: smallest $\epsilon \in (0, \epsilon\_prev]$ such that $ESS(w_j(\epsilon)) \geq target\_ess$
    with the weight update (Algorithm 5 p.24):

        $w^i_j \propto w^i_{j-1} \cdot (\sum_k 1\{ s^i_{j-1}[k] < \epsilon_j \}) / (\sum_k 1\{ s^i_{j-1}[k] < \epsilon_{j-1} \})$
    """
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
#                        SYSTEMATIC RESAMPLING
# ===========================================================================
def systematic_resample(w):
    N = w.size
    p = w / w.sum()
    positions = (np.arange(N) + np.random.uniform()) / N
    cum = np.cumsum(p)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions)


# ===========================================================================
#        ROBUST CHOLESKY (dynamic regularization + diagonal fallback)
# ===========================================================================
def safe_cholesky(cov, base_jitter=1e-10, max_jitter=1.0, n_tries=20):
    """
    Cholesky decomposition with adaptive jitter.

    1) We start with a jitter proportional to the matrix scale
       (max of the diagonal), not an absolute 1.0. This is essential when
       the particle covariance collapses in the final SMC steps:
       an absolute 1e-8 jitter could be >> max-diag and thus
       completely overwhelm the structure.
    2) If Cholesky fails, we multiply the jitter by 10 and retry.
    3) Ultimate fallback: diagonal matrix = $\sqrt{diag}$; independent
       marginal Gaussian sampling per dimension. Prevents any crash.
    """
    d = cov.shape[0]
    scale = max(float(np.diag(cov).max()), 1e-12)
    jitter = base_jitter * scale
    eye = np.eye(d)
    for _ in range(n_tries):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
            if jitter > max_jitter * scale:
                break
    # Diagonal fallback
    diag = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return np.diag(diag)


# ===========================================================================
#  MOVE STEP: MH-MCMC targeting pi_eps_j (correlated pseudo-marginal)
# ===========================================================================
def move_step(thetas, U, dist_M, eps_j, kernel_chol, rho,
              sigma_obs, sigma_mu, alpha_low, alpha_high, s_obs):
    """
    One MH iteration per particle, targeting

        $\pi_{\epsilon_j}(\theta) \propto \pi(\theta) \cdot P\{ d(s(x), s(x^*)) < \epsilon_j | \theta \}$.

    Joint proposal **($\theta^*, U^*$)** AR(1)-coupled:

        $\theta^* = \theta + L \cdot Z_\theta, \quad Z_\theta \sim N(0, I)$
        $U^* = \rho \cdot U + \sqrt{1-\rho^2} \cdot Z_U, \quad Z_U \sim N(0, I)$

    L is the robust Cholesky factor of the weighted covariance
    (Section 9 p.24: "$K_j$ Gaussian kernel with cov $2 \cdot Cov(particles)$").

    The simulator is then deterministic: pseudo-dataset = $\mu + \sigma \cdot U$.
    This couples the noise between the current particle (stored dist\_M, U)
    and the proposal, which cancels the dominant variance
    of the binary MH ratio and stabilizes the chain (cf. Deligiannidis,
    Doucet, Pitt 2018).

    Acceptance probability:
        $a = \min\{ 1, [\pi(\theta^*)/\pi(\theta)] \cdot [n\_pass^*/n\_pass] \}$
    """
    N, dim = thetas.shape

    # --- Proposal of theta via robust Cholesky ---------------------------
    Z_theta = np.random.normal(size=(N, dim))
    thetas_prop = thetas + Z_theta @ kernel_chol.T

    # --- Proposal of U coupled AR(1) (correlated PM) ------------------
    Z_U = np.random.normal(size=U.shape)
    U_prop = rho * U + np.sqrt(1.0 - rho ** 2) * Z_U

    # --- DETERMINISTIC simulation in (theta_prop, U_prop) ---------------------
    dist_prop = simulate_from_noise(thetas_prop, U_prop, sigma_obs, s_obs)

    # --- Calculation of the MH ratio ---------------------------------------------
    # n_pass_cur IS STORED from the previous step (standard
    # pseudo-marginal, Andrieu & Roberts 2009) -- it is NOT re-simulated.
    n_pass_cur = (dist_M < eps_j).sum(axis=1).astype(float)
    n_pass_prop = (dist_prop < eps_j).sum(axis=1).astype(float)

    lp_cur = log_prior(thetas, sigma_mu, alpha_low, alpha_high)
    lp_prop = log_prior(thetas_prop, sigma_mu, alpha_low, alpha_high)

    with np.errstate(divide="ignore"):
        log_ratio = (lp_prop - lp_cur
                     + np.log(np.maximum(n_pass_prop, 1e-300))
                     - np.log(np.maximum(n_pass_cur, 1e-300)))
    log_ratio = np.where(np.isfinite(lp_prop) & (n_pass_prop > 0),
                         log_ratio, -np.inf)

    accept = np.log(np.random.uniform(size=N)) < log_ratio

    # Update CONSISTENTLY (theta, U, dist) -- necessary to
    # preserve the invariance of the correlated pseudo-marginal.
    thetas_new = np.where(accept[:, None], thetas_prop, thetas)
    U_new = np.where(accept[:, None, None, None], U_prop, U)
    dist_new = np.where(accept[:, None], dist_prop, dist_M)

    n_sims = N * U.shape[1] * U.shape[2] * U.shape[3]   # Z_U draws
    return thetas_new, U_new, dist_new, int(accept.sum()), n_sims


# ===========================================================================
#                          SMC-ABC MAIN LOOP
# ===========================================================================
def smc_abc(
    x_obs,
    sigma_mu,
    sigma_obs,
    N=1000,
    T=30,
    M=1,
    alpha_ess=0.95,
    N_min_frac=0.5,
    n_move_steps=2,
    rho=0.99,
    alpha_low=-4.0,
    alpha_high=4.0,
    seed=None,
) -> ABCResult:
    """
    Adaptive SMC-ABC (Algorithm 5 p.24) with correlated pseudo-marginal.

    Parameters
    ----------
    N            : number of particles (paper p.7: 10^4 toy, 10^3 G&K)
    T            : max number of SMC steps (paper p.7: 30 toy, 500 G&K)
    M            : pseudo-datasets per particle (M $\geq$ 1, p.24)
    alpha_ess    : ESS decay factor (Del Moral 2012)
    N_min_frac   : resampling threshold, ESS < N_min_frac * N
    n_move_steps : number of MH iterations per SMC step
    rho          : AR(1) correlation on auxiliary noise U
                    (Deligiannidis, Doucet, Pitt 2018). $\rho=0$: standard
                    pseudo-marginal. $\rho \approx 1$: maximal coupling.
    """
    if seed is not None:
        np.random.seed(seed)

    n, K = x_obs.shape
    s_obs = x_obs.mean(axis=1)
    N_min = int(N_min_frac * N)

    t0 = time.perf_counter()

    # --- Initialization: particles ~ prior, U ~ N(0,I), eps_0 = max d --
    thetas = sample_prior(N, n, sigma_mu, alpha_low, alpha_high)
    U = np.random.normal(size=(N, M, n, K))
    dist_M = simulate_from_noise(thetas, U, sigma_obs, s_obs)
    eps = dist_M.max()
    w = np.ones(N)

    n_sims = N * (n + M * n * K)              # init: prior + U draws
    eps_history = [eps]
    accept_history = []

    for j in range(1, T + 1):
        # 1) Adaptive threshold selection & reweighting
        target_ess = alpha_ess * _ess(w)
        eps_new, w_new = find_next_eps(dist_M, eps, w, target_ess)
        if w_new.sum() <= 0 or eps_new >= eps:
            break
        eps = eps_new
        w = w_new

        # 2) Resample if ESS < N_min (weights and U must follow)
        if _ess(w) < N_min:
            idx = systematic_resample(w)
            thetas = thetas[idx]
            U = U[idx]
            dist_M = dist_M[idx]
            w = np.ones(N)

        # 3) Weighted cov + robust Cholesky for the K_j kernel
        wp = w / w.sum()
        mean_w = (wp[:, None] * thetas).sum(axis=0)
        diff = thetas - mean_w
        cov_w = (wp[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(0)
        # 2 * Cov (Section 9 p.24) + Cholesky with dynamic jitter
        kernel_chol = safe_cholesky(2.0 * cov_w)

        # 4) MH Move(s) with correlated pseudo-marginal
        n_acc_total = 0
        for _ in range(n_move_steps):
            thetas, U, dist_M, n_acc, ns = move_step(
                thetas, U, dist_M, eps, kernel_chol, rho,
                sigma_obs, sigma_mu, alpha_low, alpha_high, s_obs,
            )
            n_acc_total += n_acc
            n_sims += ns

        eps_history.append(eps)
        accept_history.append(n_acc_total / (N * n_move_steps))

    # Final resampling for uniformly weighted iid sample
    idx = systematic_resample(w)
    thetas_final = thetas[idx]
    cpu_time = time.perf_counter() - t0

    return ABCResult(
        alpha_chain=thetas_final[:, 0],
        mu_chain=thetas_final[:, 1:],
        accepted_dists_mu=np.asarray(accept_history),    # acceptance rate
        accepted_dists_alpha=np.asarray(eps_history),    # eps_j trajectory
        cpu_time=cpu_time,
        n_model_sims=n_sims,
        config=dict(
            algorithm="SMC-ABC",
            N=N, T=T, M=M, alpha_ess=alpha_ess, N_min_frac=N_min_frac,
            n_move_steps=n_move_steps, rho=rho, n=n, K=K,
            sigma_mu=sigma_mu, sigma_obs=sigma_obs, seed=seed,
            n_steps_done=len(eps_history) - 1,
            final_eps=float(eps_history[-1]),
            mean_accept_rate=float(np.mean(accept_history))
            if accept_history else float("nan"),
        ),
    )