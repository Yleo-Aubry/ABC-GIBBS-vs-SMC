"""
============================================================================
  data_collection.py
  --------------------------------------------------------------------------
  Lance R=10 réplicats indépendants pour chaque méthode, sur chaque modèle,
  et pickle les sorties brutes nécessaires aux figures dans `cache/`.

  Aucune modification de la logique des algorithmes : on appelle simplement
  `abc_gibbs`, `smc_abc`, `abc_gibbs_gk`, `smc_abc_gk` avec leurs `seed=`,
  puis on stocke (alpha_chain, mu_chain, eps_history, n_model_sims, ...).

  Usage:
      from data_collection import collect_all
      collect_all(R=10, base_seed=100, force=False)

  Avec `force=False`, on ne relance les sims que si le pickle est absent.
============================================================================
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from abc_gibbs import abc_gibbs
from smc_abc import smc_abc
from comparison import (
    true_posterior_normal_normal,
    inferential_error,
)
from gk_model import simulate_gk
from abc_gibbs_gk import abc_gibbs_gk
from smc_abc_gk import smc_abc_gk
from comparison_gk import posterior_predictive_distance


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------------------------------
def _path(name): return os.path.join(CACHE_DIR, name)


def _stack_results(results, with_eps=False):
    """Empile les chaînes alpha (R, T) et mu (R, T, n) sur la liste results."""
    alpha = np.stack([r.alpha_chain for r in results], axis=0)
    mu = np.stack([r.mu_chain for r in results], axis=0)
    n_sims = np.array([r.n_model_sims for r in results])
    cpu = np.array([r.cpu_time for r in results])
    out = dict(alpha=alpha, mu=mu, n_sims=n_sims, cpu=cpu)
    if with_eps:
        # SMC-ABC range eps_history dans `accepted_dists_alpha`.
        out["eps"] = [np.asarray(r.accepted_dists_alpha) for r in results]
        out["accept_rate"] = [np.asarray(r.accepted_dists_mu) for r in results]
    return out


# ==========================================================================
#                      MODÈLE NORMAL-NORMAL  (Section 3.2)
# ==========================================================================
def collect_nn(R=10, base_seed=100, force=False):
    out_path = _path("nn.pkl")
    if not force and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)

    sigma_obs, sigma_mu = 1.0, 1.0
    n, K = 20, 10
    rng = np.random.default_rng(42)
    alpha_true = 2.5
    mu_true = rng.normal(alpha_true, sigma_mu, size=n)
    x_obs = rng.normal(mu_true[:, None], sigma_obs, size=(n, K))

    truth = true_posterior_normal_normal(x_obs, sigma_mu, sigma_obs)

    res_g, res_s = [], []
    inf_g, inf_s = [], []
    for r in range(R):
        rg = abc_gibbs(x_obs, sigma_mu, sigma_obs,
                       N_iter=1000, N_mu=30, N_alpha=30, seed=base_seed + r)
        rs = smc_abc(x_obs, sigma_mu, sigma_obs,
                     N=1000, T=30, M=1, alpha_ess=0.95,
                     N_min_frac=0.5, n_move_steps=2, seed=base_seed + r)
        res_g.append(rg); res_s.append(rs)
        inf_g.append(inferential_error(rg, truth))
        inf_s.append(inferential_error(rs, truth))

    pack = dict(
        x_obs=x_obs, alpha_true=alpha_true, mu_true=mu_true,
        sigma_obs=sigma_obs, sigma_mu=sigma_mu, n=n, K=K,
        truth=truth,
        gibbs=_stack_results(res_g, with_eps=False),
        smc=_stack_results(res_s, with_eps=True),
        inf_gibbs=inf_g, inf_smc=inf_s,
    )
    pack["gibbs"]["wass_alpha"] = np.array([d["wass_alpha"] for d in inf_g])
    pack["gibbs"]["mean_wass_mu"] = np.array([d["mean_wass_mu"] for d in inf_g])
    pack["smc"]["wass_alpha"] = np.array([d["wass_alpha"] for d in inf_s])
    pack["smc"]["mean_wass_mu"] = np.array([d["mean_wass_mu"] for d in inf_s])

    with open(out_path, "wb") as f:
        pickle.dump(pack, f)
    return pack


# ==========================================================================
#                          MODÈLE G & K  (Section 4)
# ==========================================================================
def collect_gk(R=10, base_seed=100, force=False):
    out_path = _path("gk.pkl")
    if not force and os.path.exists(out_path):
        with open(out_path, "rb") as f:
            return pickle.load(f)

    n, K = 50, 100
    B, g, k = 1.0, 0.2, 0.5
    sigma_mu = 1.0
    alpha_true = 3.0
    rng = np.random.default_rng(42)
    mu_true = rng.normal(alpha_true, sigma_mu, size=n)
    z_obs = rng.standard_normal(size=(n, K))
    x_obs = simulate_gk(mu_true[:, None], B, g, k, z=z_obs)

    res_g, res_s = [], []
    pp_g, pp_s = [], []
    for r in range(R):
        rg = abc_gibbs_gk(
            x_obs, sigma_mu=sigma_mu, B=B, g=g, k=k,
            N_iter=300, N_mu=30, N_alpha=30, burn_in=10, seed=base_seed + r,
        )
        rs = smc_abc_gk(
            x_obs, B=B, g=g, k=k,
            N=500, T=20, M=1, alpha_ess=0.95,
            N_min_frac=0.5, n_move_steps=2, rho=0.99, seed=base_seed + r,
        )
        res_g.append(rg); res_s.append(rs)
        pp_g.append(posterior_predictive_distance(rg, x_obs, B, g, k,
                                                  seed=base_seed + r))
        pp_s.append(posterior_predictive_distance(rs, x_obs, B, g, k,
                                                  seed=base_seed + r))

    pack = dict(
        x_obs=x_obs, alpha_true=alpha_true, mu_true=mu_true,
        B=B, g=g, k=k, sigma_mu=sigma_mu, n=n, K=K,
        gibbs=_stack_results(res_g, with_eps=False),
        smc=_stack_results(res_s, with_eps=True),
        pp_gibbs=pp_g, pp_smc=pp_s,
    )
    pack["gibbs"]["pred_mean"] = np.array([d["mean_pred_dist"] for d in pp_g])
    pack["smc"]["pred_mean"] = np.array([d["mean_pred_dist"] for d in pp_s])

    with open(out_path, "wb") as f:
        pickle.dump(pack, f)
    return pack


def collect_all(R=10, base_seed=100, force=False):
    nn = collect_nn(R=R, base_seed=base_seed, force=force)
    gk = collect_gk(R=R, base_seed=base_seed, force=force)
    return nn, gk


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    collect_all(R=args.R, base_seed=args.seed, force=args.force)
    print("[ok] cache écrit dans", CACHE_DIR)
