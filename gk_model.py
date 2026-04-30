"""
============================================================================
   G & K model and octile statistics
============================================================================

G & K distribution (Tukey 1977 ; Rayner & MacGillivray 2002): family of
flexible distributions ONLY defined by their inverse quantile function. The
density has no closed form -> strictly intractable likelihood,
which makes it a canonical benchmark for ABC methods (cf. Prangle
2017 ; Fearnhead & Prangle 2012 ; Clarté et al. 2020 Section 4).

Quantile function (Clarté et al. 2020 paper, Section 4, p.7):

   F^{-1}(u; mu, B, g, k, c) =
       mu + B * ( 1 + c * (1 - exp(-g*z)) / (1 + exp(-g*z)) )
                 * (1 + z^2)^k * z,         with z = Phi^{-1}(u),

   c = 0.8 by convention (Prangle 2017).

References
----------
 * Tukey (1977) "Modern Techniques in Data Analysis."
 * Rayner & MacGillivray (2002) "Numerical maximum likelihood estimation
        for the g-and-k and generalized g-and-h distributions."
        Statist. Comput. 12, 57-75.
 * Prangle (2017) "gk: An R package for the g-and-k and generalised
        g-and-h distributions."  arXiv:1706.06889.
 * Fearnhead & Prangle (2012) "Constructing summary statistics for
        approximate Bayesian computation: semi-automatic ABC."  JRSS-B.
 * Clarté, Robert, Ryder, Stoehr (2020) Section 4 (hierarchical model).

Retained summary statistic (paper eq. p.7): OCTILES + MAE distance
on the 9 quantiles {0, 1/8, 2/8, ..., 1}. The empirical mean is
not sufficient here -- the choice of an octile statistic is motivated
by its robustness to the heavy tails typical of the G&K distribution.
============================================================================
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ===========================================================================
#                            G & K SIMULATOR
# ===========================================================================
def simulate_gk(mu, B=1.0, g=0.2, k=0.5, size=None, c=0.8, U=None, z=None):
    """
    Vectorized G&K sampler via the inverse quantile function.

    Standard path (user spec): draw U ~ U(0,1) and apply
    F^{-1}(U). Alternative path (useful for the Correlated Pseudo-Marginal
    of SMC-ABC): directly inject the corresponding auxiliary variables
    `z ~ N(0,1)` (saves a call to `norm.ppf`).

    Parameters
    ----------
    mu        : scalar or ndarray (broadcasted with z/U)
    B, g, k   : G&K parameters. Defaults to B=1, g=0.2, k=0.5
                (standard setting from Section 4 of the paper).
    size      : sample shape if U / z are not provided.
    c         : 0.8 (Prangle 2017).
    U         : (optional) draws U ~ U(0,1), shape `size`.
    z         : (optional) precomputed draws z ~ N(0,1).
                If provided, bypasses U and `norm.ppf`.

    Returns
    -------
    x : ndarray of the same shape as `mu` broadcasted with `z`.
    """
    if z is None:
        if U is None:
            U = np.random.uniform(size=size)
        z = norm.ppf(U)
    e = np.exp(-g * z)
    skew = 1.0 + c * (1.0 - e) / (1.0 + e)        # skewness driven by g
    kurt = (1.0 + z ** 2) ** k                    # kurtosis driven by k
    return mu + B * skew * kurt * z


# ===========================================================================
#                OCTILE STATISTICS + DISTANCE  (vectorized)
# ===========================================================================
_OCTILE_PROBS = np.linspace(0.0, 1.0, 9)          # 0, 1/8, ..., 1


def octiles(x, axis=-1):
    """
    Computes the 9 octiles of `x` along `axis`. Returns the result
    with the quantiles axis moved to the LAST position: if x has
    shape (..., K), the output is (..., 9). Convenient for summations.
    """
    q = np.quantile(x, _OCTILE_PROBS, axis=axis)
    return np.moveaxis(q, 0, -1)


def octile_distance(x_sim, x_obs):
    """
    MAE distance on the 9 octiles (paper Section 4, p.7):

        d(x_1, x_2) = sum_{j=0..8} |q(x_1, j/8) - q(x_2, j/8)|.

    Expected shape
    --------------
    x_sim : shape (..., K)    -- one or more simulated series
    x_obs : shape (K,) or (..., K)
            if shape (K,), broadcast over the prefix of x_sim.

    Returns
    -------
    distance : shape (...)    a scalar distance per series.
    """
    q_sim = octiles(x_sim, axis=-1)               # (..., 9)
    q_obs = octiles(x_obs, axis=-1)               # (..., 9) or (9,)
    return np.sum(np.abs(q_sim - q_obs), axis=-1)


def grouped_octile_distance(x_sim, q_obs_per_group):
    """
    Distance for the JOINT version (used by SMC-ABC):
    sum of the octile distances over the n groups.

        d(x, x*) = sum_i sum_{j=0..8} |q(x_i, j/8) - q(x_i*, j/8)|.

    Parameters
    ----------
    x_sim            : (..., n, K)
    q_obs_per_group  : (n, 9)   precomputed octiles of the observations.

    Returns
    -------
    distance : (...)
    """
    q_sim = octiles(x_sim, axis=-1)                             # (..., n, 9)
    return np.sum(np.abs(q_sim - q_obs_per_group), axis=(-1, -2))