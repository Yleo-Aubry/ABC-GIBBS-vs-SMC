"""
============================================================================
   EXACT Random Walk Metropolis (Adaptive Metropolis)
   for the hierarchical Normal-Normal model -- dynamic ground truth
============================================================================

This sampler draws from the TRUE posterior (no ABC approximation),
by exploiting the closed-form Gaussian log-likelihood of the model:

    alpha ~ U[-4, 4]
    mu_j | alpha ~ N(alpha, sigma_mu^2)            j = 1..n
    x_{jk} | mu_j ~ N(mu_j, sigma_obs^2)           k = 1..K

It serves as a **dynamic reference** against ABC-Gibbs and SMC-ABC: its
posterior sample is ν_0 (the target that the pseudo-posteriors ν_ε
of the ABC algorithms are supposed to approach when ε -> 0, cf. paper
Theorem 3 and Section 6).

----------------------------------------------------------------------------
References:

  * Metropolis, Rosenbluth, Rosenbluth, Teller, Teller (1953)
        "Equation of State Calculations by Fast Computing Machines."
        J. Chem. Phys. 21(6), 1087-1092.
        --> original algorithm.

  * Hastings (1970)
        "Monte Carlo sampling methods using Markov chains and their
         applications."  Biometrika 57(1), 97-109.
        --> generalization to non-symmetric kernels.

  * Roberts, Gelman, Gilks (1997)
        "Weak convergence and optimal scaling of random walk Metropolis
         algorithms."  Ann. Appl. Probab. 7(1), 110-120.
        --> optimal scaling factor 2.38^2 / d in dimension d
            for Gaussian targets (target acceptance rate ~0.234).

  * Haario, Saksman, Tamminen (2001)
        "An adaptive Metropolis algorithm."  Bernoulli 7(2), 223-242.
        --> ADAPTATION OF THE PROPOSAL COVARIANCE during the
            burn-in phase: Sigma_t = empirical Cov of past
            iterations, allowing the chain to learn the correlation
            between alpha and the mu_j's (which is strong here due to the hierarchy).

  * Andrieu, Thoms (2008)
        "A tutorial on adaptive MCMC."  Statist. Comput. 18, 343-373.
        --> modern review and controlled adaptation schemes.

----------------------------------------------------------------------------
Retained adaptation strategy (Haario et al. 2001)
---------------------------------------------------

1) During the first B iterations (burn-in), we update the empirical mean
   and covariance `Sigma_t` of past iterations at each step via Welford's
   stable recurrence. The proposal is:

        theta* ~ N(theta_t, (2.38^2 / d) * Sigma_t + epsilon_reg * I)

   where the epsilon_reg * I term (Haario et al. 2001) guarantees ergodicity
   and prevents the collapse of Sigma_t in the early steps.

2) After burn-in, Sigma is FROZEN at its last value. The chain then
   restarts as a standard RWM with a fixed kernel, which simplifies
   theoretical analysis (the retained sample is a true reversible
   homogeneous Markov chain).

3) We use a Cholesky decomposition `L` of the frozen covariance
   to sample the proposals via theta* = theta + L * Z, which is faster
   and more stable than `np.random.multivariate_normal` (which recomputes
   its own Cholesky at each call).
============================================================================
"""

from __future__ import annotations

import time

import numpy as np

from abc_gibbs import ABCResult


# ===========================================================================
#                EXACT LOG-POSTERIOR  (vectorized)
# ===========================================================================
def log_posterior(theta, x_obs, sigma_mu, sigma_obs,
                  alpha_low=-4.0, alpha_high=4.0):
    """
    log P(theta | x) = log pi(alpha) + sum_j log pi(mu_j | alpha)
                                     + sum_{j,k} log f(x_{jk} | mu_j)

    Vectorized: no Python loops over j or k. Accepts theta of
    shape (d,) or (B, d) with d = n+1 (useful to evaluate multiple
    candidates in parallel if ever needed -- here we use it
    in (d,) mode).

    Parameters
    ----------
    theta : array (d,) or (B, d)
        theta[..., 0] = alpha,  theta[..., 1:] = mu_1..mu_n
    x_obs : array (n, K)
    sigma_mu, sigma_obs : floats
    alpha_low, alpha_high : bounds of the uniform prior on alpha

    Returns
    -------
    log_p : float or array (B,) -- -inf if alpha is outside [-4, 4].
    """
    theta = np.asarray(theta)
    a = theta[..., 0]
    mu = theta[..., 1:]                      # (..., n)

    # 1) prior pi(alpha) = U[-4, 4]    (log(8) constant omitted)
    inside = (a >= alpha_low) & (a <= alpha_high)

    # 2) conditional prior pi(mu_j | alpha) = N(alpha, sigma_mu^2)
    #    sum_j log phi(mu_j; alpha, sigma_mu^2) -- vectorized over axis -1
    n = mu.shape[-1]
    log_p_mu_given_a = (
        -0.5 * np.sum((mu - a[..., None]) ** 2, axis=-1) / sigma_mu ** 2
        - 0.5 * n * np.log(2.0 * np.pi * sigma_mu ** 2)
    )

    # 3) likelihood f(x_{jk} | mu_j) = N(mu_j, sigma_obs^2)
    #    sum_j sum_k log phi(x_{jk}; mu_j, sigma_obs^2). We first sum
    #    over k (axis 1 of x_obs), then over j (axis -1 of the result).
    #    Broadcast: x_obs (n, K), mu (..., n) -> (..., n, K)
    K = x_obs.shape[1]
    diff = x_obs - mu[..., :, None]                              # (..., n, K)
    log_lik = (
        -0.5 * np.sum(diff ** 2, axis=(-1, -2)) / sigma_obs ** 2
        - 0.5 * n * K * np.log(2.0 * np.pi * sigma_obs ** 2)
    )

    log_p = log_p_mu_given_a + log_lik
    # Strict enforcement of the uniform prior support on alpha.
    return np.where(inside, log_p, -np.inf)


# ===========================================================================
#        ROBUST CHOLESKY (cf. same principles as in smc_abc.py)
# ===========================================================================
def _safe_cholesky(cov, base_jitter=1e-10, n_tries=20):
    """Cholesky with dynamic jitter scaled to the matrix magnitude."""
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
#                ALGORITHM: Adaptive Random Walk Metropolis
# ===========================================================================
def random_walk_metropolis(
    x_obs,
    sigma_mu,
    sigma_obs,
    N_iter=10000,
    burn_in=2000,
    alpha_init=0.0,
    mu_init=None,
    alpha_low=-4.0,
    alpha_high=4.0,
    epsilon_reg=1e-8,
    adapt_start=50,
    seed=None,
) -> ABCResult:
    """
    Exact RWM + adaptation from Haario, Saksman, Tamminen (2001).

    Steps
    ------
    a) Adaptation phase (iterations 0..burn_in-1):
       - At each step, recursive update of the empirical mean and covariance
         of the chain via Welford's formula (numerically stable, without
         recomputing the whole chain).
       - From `adapt_start` (~50) onwards, we use Sigma_t to
         build the proposal covariance:

             prop_cov = (2.38^2 / d) * Sigma_t  +  epsilon_reg * I

         where the 2.38^2/d factor is the asymptotic optimum from Roberts,
         Gelman, Gilks (1997), and the epsilon_reg*I term ensures
         ergodicity (Haario et al. 2001, Theorem 1).
       - Before `adapt_start`, we use an arbitrary diagonal covariance
         to start (otherwise Sigma_0 is singular).

    b) Stationary phase (iterations burn_in..N_iter-1):
       - Sigma is FROZEN at its final value, prop_cov is fixed and
         Cholesky-decomposed once and for all -> homogeneous chain.

    Returns: ABCResult with:
        alpha_chain          (N_iter - burn_in,)
        mu_chain             (N_iter - burn_in, n)
        accepted_dists_mu    rolling acceptance rate (window 100)
        accepted_dists_alpha empty array (not relevant in joint RWM)
        cpu_time, n_model_sims (= nb of log_posterior evals), config.
    """
    if seed is not None:
        np.random.seed(seed)

    n, K = x_obs.shape
    d = n + 1                                     # dimension of theta

    # ----- Initial state ---------------------------------------------------
    if mu_init is None:
        # "reasonable" init: empirical means per group (MLE of mu_j)
        mu_init = x_obs.mean(axis=1)
    theta_cur = np.concatenate([[alpha_init], np.asarray(mu_init, float)])
    log_p_cur = log_posterior(theta_cur, x_obs, sigma_mu, sigma_obs,
                              alpha_low, alpha_high)
    n_evals = 1                                   # 1 initial eval

    # ----- Storage -------------------------------------------------------
    chain = np.empty((N_iter, d))
    accept_flags = np.empty(N_iter, dtype=bool)

    # ----- Welford initialization for adaptation -------------------------
    # mean_t, M2_t accumulate the mean and the sum of centered squares.
    # We start with theta_cur to bootstrap.
    mean_t = theta_cur.copy()
    M2_t = np.zeros((d, d))
    count_t = 1

    # Diagonal covariance to start (before adapt_start): we calibrate it
    # on the typical orders of magnitude of the model.
    init_diag = np.full(d, (0.1 * sigma_mu) ** 2)
    init_diag[0] = (0.1 * (alpha_high - alpha_low)) ** 2     # alpha
    prop_cov = np.diag(init_diag)
    prop_chol = _safe_cholesky(prop_cov)

    scale = (2.38 ** 2) / d                       # Roberts, Gelman, Gilks 1997

    t0 = time.perf_counter()

    # ====================================================================
    #                MAIN LOOP (burn-in + stationary)
    # ====================================================================
    for t in range(N_iter):
        # 1) Multivariate proposal: theta* = theta + L * Z
        z = np.random.normal(size=d)
        theta_prop = theta_cur + prop_chol @ z

        # 2) Log-posterior at the proposal
        log_p_prop = log_posterior(theta_prop, x_obs, sigma_mu, sigma_obs,
                                   alpha_low, alpha_high)
        n_evals += 1

        # 3) MH acceptance (symmetric proposal: ratio = exp(Δ logπ))
        log_u = np.log(np.random.uniform())
        if log_u < (log_p_prop - log_p_cur):
            theta_cur = theta_prop
            log_p_cur = log_p_prop
            accept_flags[t] = True
        else:
            accept_flags[t] = False

        chain[t] = theta_cur

        # 4) Welford adaptation of the empirical covariance
        #    (cf. Welford 1962 "Note on a method for calculating corrected
        #     sums of squares and products", Technometrics 4(3).)
        if t < burn_in:
            count_t += 1
            delta = theta_cur - mean_t
            mean_t = mean_t + delta / count_t
            delta2 = theta_cur - mean_t
            M2_t = M2_t + np.outer(delta, delta2)         # update SPD-safe

            # We start using the empirical covariance only
            # after `adapt_start` iterations, otherwise Sigma_t is too noisy.
            if count_t > adapt_start:
                Sigma_t = M2_t / (count_t - 1)
                prop_cov = scale * Sigma_t + epsilon_reg * np.eye(d)
                prop_chol = _safe_cholesky(prop_cov)
        elif t == burn_in:
            # COVARIANCE FREEZE: we freeze prop_chol for the stationary
            # phase. The chain becomes homogeneous and therefore strictly
            # reversible Markovian (Haario et al. 2001, standard
            # conditions for the diminishing adaptation criterion).
            pass

    cpu_time = time.perf_counter() - t0

    # ----- Result formatting -------------------------------------
    post = chain[burn_in:]
    alpha_chain = post[:, 0]
    mu_chain = post[:, 1:]

    # Rolling acceptance rate (window 100, post burn-in)
    win = 100
    if post.shape[0] >= win:
        kernel = np.ones(win) / win
        rolling_acc = np.convolve(
            accept_flags[burn_in:].astype(float), kernel, mode="valid"
        )
    else:
        rolling_acc = np.array([accept_flags[burn_in:].mean()])

    return ABCResult(
        alpha_chain=alpha_chain,
        mu_chain=mu_chain,
        accepted_dists_mu=rolling_acc,             # acceptance rate
        accepted_dists_alpha=np.array([]),         # not relevant
        cpu_time=cpu_time,
        n_model_sims=n_evals,
        config=dict(
            algorithm="RWM-AM",
            N_iter=N_iter, burn_in=burn_in, n=n, K=K, d=d,
            sigma_mu=sigma_mu, sigma_obs=sigma_obs,
            alpha_low=alpha_low, alpha_high=alpha_high,
            epsilon_reg=epsilon_reg, adapt_start=adapt_start,
            scale_factor=scale, seed=seed,
            overall_accept_rate=float(accept_flags.mean()),
            stationary_accept_rate=float(accept_flags[burn_in:].mean()),
            final_prop_cov=prop_cov,
        ),
    )