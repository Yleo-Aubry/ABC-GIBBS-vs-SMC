"""
============================================================================
  figures.py
  --------------------------------------------------------------------------
  Génère toutes les figures du rapport sur le Projet 4 "Gibbs and ABC"
  (Clarté et al. 2021), à partir des sorties picklées par data_collection.py.

  Style cohérent avec un document LaTeX (font serif, axes 10pt).
  Chaque figure est sauvegardée en deux versions :
      figures/<nom>_color.pdf   -- palette colorblind-friendly
      figures/<nom>_bw.pdf      -- niveaux de gris pour impression N&B

  Aucun titre dans la figure : seulement labels d'axes + légende.
  Les algorithmes ne sont JAMAIS rappelés ici ; on lit le pickle.
============================================================================
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ==========================================================================
#                         STYLE GLOBAL  +  PALETTES
# ==========================================================================
def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


PALETTE_COLOR = {
    "gibbs": "#1f4e79",
    "smc":   "#c0504d",
    "truth": "black",
    "ref":   "black",
}
PALETTE_BW = {
    "gibbs": "#222222",
    "smc":   "#888888",
    "truth": "black",
    "ref":   "black",
}
LINESTYLES_BW = {"gibbs": "-", "smc": "--", "truth": "-", "ref": ":"}
HATCH_BW = {"gibbs": "////", "smc": "...."}


def _palette(bw): return PALETTE_BW if bw else PALETTE_COLOR
def _ls(bw, key): return LINESTYLES_BW[key] if bw else "-"


def _save(fig, name, bw):
    suffix = "_bw" if bw else "_color"
    fig.savefig(os.path.join(FIG_DIR, f"{name}{suffix}.pdf"))
    plt.close(fig)


def _both(fn):
    """Decorateur : appelle fn(bw=False) puis fn(bw=True) -> 2 PDFs."""
    def wrapped(*args, **kwargs):
        for bw in (False, True):
            fn(*args, bw=bw, **kwargs)
    return wrapped


# ==========================================================================
#                            UTILITAIRES
# ==========================================================================
def _flatten(arr):
    """Aplatit (R, T) ou (R, T, n) sur la dimension batch."""
    return np.asarray(arr).reshape(-1, *np.asarray(arr).shape[2:])


def _box_style(bp, color, bw, hatch_key=None):
    for el in ("boxes", "whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set_color(color)
    for box in bp["boxes"]:
        box.set_linewidth(1.2)
    if bw and hatch_key is not None:
        for box in bp["boxes"]:
            box.set_hatch(HATCH_BW[hatch_key])
    for med in bp["medians"]:
        med.set_linewidth(1.5)


# ==========================================================================
#  FIG 1 -- Boxplots W1(alpha) et W1(mu_j) -- Normal-Normal
# ==========================================================================
@_both
def fig_boxplot_W1_NN(nn, bw=False):
    pal = _palette(bw)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # -- Panel alpha --
    data_a = [nn["gibbs"]["wass_alpha"], nn["smc"]["wass_alpha"]]
    bp1 = axes[0].boxplot(data_a, positions=[1, 2], widths=0.55,
                          patch_artist=True)
    for box, key in zip(bp1["boxes"], ("gibbs", "smc")):
        box.set_facecolor("none" if bw else pal[key])
        box.set_alpha(1.0 if bw else 0.55)
        box.set_edgecolor(pal[key])
    _box_style(bp1, pal["gibbs"], bw, "gibbs")  # not perfect but readable
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(["ABC-Gibbs", "SMC-ABC"])
    axes[0].set_ylabel(r"$W_1(\alpha)$")

    # -- Panel mu --
    data_m = [nn["gibbs"]["mean_wass_mu"], nn["smc"]["mean_wass_mu"]]
    bp2 = axes[1].boxplot(data_m, positions=[1, 2], widths=0.55,
                          patch_artist=True)
    for box, key in zip(bp2["boxes"], ("gibbs", "smc")):
        box.set_facecolor("none" if bw else pal[key])
        box.set_alpha(1.0 if bw else 0.55)
        box.set_edgecolor(pal[key])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(["ABC-Gibbs", "SMC-ABC"])
    axes[1].set_ylabel(r"$\overline{W_1}(\mu_j)$")

    # Échelle log si SMC-ABC écrase visuellement ABC-Gibbs
    ratio = np.median(data_m[1]) / max(np.median(data_m[0]), 1e-12)
    if ratio > 5 or ratio < 0.2:
        axes[1].set_yscale("log")

    _save(fig, "fig_boxplot_W1_NN", bw)


# ==========================================================================
#  FIG 2 -- Postérieure de alpha (NN) avec densité analytique
# ==========================================================================
@_both
def fig_posterior_alpha_NN(nn, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))

    a_g = _flatten(nn["gibbs"]["alpha"])
    a_s = _flatten(nn["smc"]["alpha"])

    ax.hist(a_g, bins=50, density=True, alpha=0.45 if not bw else 0.6,
            color=pal["gibbs"], edgecolor=pal["gibbs"],
            histtype="stepfilled" if not bw else "step", linewidth=1.2,
            hatch=HATCH_BW["gibbs"] if bw else None,
            label="ABC-Gibbs")
    ax.hist(a_s, bins=50, density=True, alpha=0.45 if not bw else 0.6,
            color=pal["smc"], edgecolor=pal["smc"],
            histtype="stepfilled" if not bw else "step", linewidth=1.2,
            hatch=HATCH_BW["smc"] if bw else None,
            label="SMC-ABC")

    truth = nn["truth"]
    ax.plot(truth["alpha_grid"], truth["alpha_pdf"],
            color=pal["truth"], linewidth=1.6, linestyle="-",
            label="analytique")

    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("densité")
    ax.set_xlim(-4, 4)
    ax.legend(loc="upper left", frameon=False)
    _save(fig, "fig_posterior_alpha_NN", bw)


# ==========================================================================
#  FIG 3 -- Postérieures conditionnelles mu_j (grille 2x3)
# ==========================================================================
@_both
def fig_posterior_mu_NN(nn, bw=False):
    pal = _palette(bw)
    js = [1, 5, 10, 15, 20]   # indices 1-based
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), sharex=False)
    axes = axes.ravel()

    sigma_mu = nn["sigma_mu"]; sigma_obs = nn["sigma_obs"]; K = nn["K"]
    x_obs = nn["x_obs"]
    a_g_mean = _flatten(nn["gibbs"]["alpha"]).mean()

    # Postérieure analytique conditionnelle (Prop. 3.1 eq.1)
    s2 = 1.0 / (1.0 / sigma_mu**2 + K / sigma_obs**2)

    mu_g = _flatten(nn["gibbs"]["mu"])     # (R*T, n)
    mu_s = _flatten(nn["smc"]["mu"])

    for ax, j1 in zip(axes, js):
        j = j1 - 1
        ax.hist(mu_g[:, j], bins=40, density=True,
                color=pal["gibbs"], alpha=0.45 if not bw else 0.6,
                histtype="stepfilled" if not bw else "step",
                hatch=HATCH_BW["gibbs"] if bw else None,
                edgecolor=pal["gibbs"], linewidth=1.0,
                label="ABC-Gibbs")
        ax.hist(mu_s[:, j], bins=40, density=True,
                color=pal["smc"], alpha=0.45 if not bw else 0.6,
                histtype="stepfilled" if not bw else "step",
                hatch=HATCH_BW["smc"] if bw else None,
                edgecolor=pal["smc"], linewidth=1.0,
                label="SMC-ABC")

        m_j = s2 * (a_g_mean / sigma_mu**2 + x_obs[j].sum() / sigma_obs**2)
        xs = np.linspace(m_j - 4*np.sqrt(s2), m_j + 4*np.sqrt(s2), 400)
        pdf = np.exp(-0.5 * (xs - m_j)**2 / s2) / np.sqrt(2*np.pi*s2)
        ax.plot(xs, pdf, color=pal["truth"], linewidth=1.4,
                label="analytique")

        ax.set_xlabel(rf"$\mu_{{{j1}}}$")
        ax.set_ylabel("densité")

    axes[-1].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_posterior_mu_NN", bw)


# ==========================================================================
#  FIG 4 -- Traceplots ABC-Gibbs (alpha et mu_1) -- 1 réplicat
# ==========================================================================
@_both
def fig_traceplots_NN(nn, bw=False):
    pal = _palette(bw)
    fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

    a = nn["gibbs"]["alpha"][0]    # premier réplicat
    mu1 = nn["gibbs"]["mu"][0, :, 0]
    a_ref = nn["truth"]["alpha_mean"]
    mu1_ref = nn["truth"]["mu_mean"][0]

    axes[0].plot(a, color=pal["gibbs"], linewidth=0.7)
    axes[0].axhline(a_ref, color=pal["ref"], linestyle="--", linewidth=1.0,
                    label="postérieure analytique (mean)")
    axes[0].set_ylabel(r"$\alpha$")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(mu1, color=pal["gibbs"], linewidth=0.7)
    axes[1].axhline(mu1_ref, color=pal["ref"], linestyle="--", linewidth=1.0)
    axes[1].set_ylabel(r"$\mu_1$")
    axes[1].set_xlabel("itération $i$")
    fig.tight_layout()
    _save(fig, "fig_traceplots_NN", bw)


# ==========================================================================
#  FIG 5 -- Calendrier eps_t SMC-ABC (NN)
# ==========================================================================
@_both
def fig_eps_schedule_NN(nn, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, eps in enumerate(nn["smc"]["eps"]):
        ax.plot(np.arange(len(eps)), eps,
                color=pal["smc"],
                alpha=0.35 if r > 0 else 0.9,
                linewidth=0.9,
                label="SMC-ABC" if r == 0 else None)
    # seuil final = médiane sur réplicats
    final_eps = np.array([eps[-1] for eps in nn["smc"]["eps"]])
    ax.axhline(np.median(final_eps), color=pal["ref"],
               linestyle="--", linewidth=1.0,
               label=fr"$\epsilon_T$ médian = {np.median(final_eps):.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("étape SMC $t$")
    ax.set_ylabel(r"$\epsilon_t$")
    ax.legend(frameon=False)
    _save(fig, "fig_eps_schedule_NN", bw)


# ==========================================================================
#  FIG 6 -- Boxplot distance prédictive G&K
# ==========================================================================
@_both
def fig_boxplot_predictive_GK(gk, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [gk["gibbs"]["pred_mean"], gk["smc"]["pred_mean"]]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.55, patch_artist=True)
    for box, key in zip(bp["boxes"], ("gibbs", "smc")):
        box.set_facecolor("none" if bw else pal[key])
        box.set_alpha(1.0 if bw else 0.55)
        box.set_edgecolor(pal[key])
        if bw:
            box.set_hatch(HATCH_BW[key])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ABC-Gibbs", "SMC-ABC"])
    ax.set_ylabel("distance prédictive (somme MAE octiles)")
    _save(fig, "fig_boxplot_predictive_GK", bw)


# ==========================================================================
#  FIG 7 -- Postérieure de alpha (G&K) avec ligne alpha*=3
# ==========================================================================
@_both
def fig_posterior_alpha_GK(gk, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))
    a_g = _flatten(gk["gibbs"]["alpha"])
    a_s = _flatten(gk["smc"]["alpha"])
    ax.hist(a_g, bins=50, density=True, alpha=0.45 if not bw else 0.6,
            color=pal["gibbs"], edgecolor=pal["gibbs"],
            histtype="stepfilled" if not bw else "step", linewidth=1.2,
            hatch=HATCH_BW["gibbs"] if bw else None, label="ABC-Gibbs")
    ax.hist(a_s, bins=50, density=True, alpha=0.45 if not bw else 0.6,
            color=pal["smc"], edgecolor=pal["smc"],
            histtype="stepfilled" if not bw else "step", linewidth=1.2,
            hatch=HATCH_BW["smc"] if bw else None, label="SMC-ABC")
    ax.axvline(gk["alpha_true"], color=pal["ref"], linestyle="--",
               linewidth=1.0, label=r"$\alpha^\star=3$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("densité")
    ax.legend(frameon=False)
    # NB : si SMC-ABC est très concentré sur une valeur loin de alpha*,
    # une alternative consiste à passer en KDE log-densité ;
    # je laisse l'histogramme pour rester directement comparable au papier.
    _save(fig, "fig_posterior_alpha_GK", bw)


# ==========================================================================
#  FIG 8 -- Calendrier eps_t SMC-ABC (G&K)
# ==========================================================================
@_both
def fig_eps_schedule_GK(gk, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r, eps in enumerate(gk["smc"]["eps"]):
        ax.plot(np.arange(len(eps)), eps,
                color=pal["smc"],
                alpha=0.35 if r > 0 else 0.9,
                linewidth=0.9,
                label="SMC-ABC" if r == 0 else None)
    final_eps = np.array([eps[-1] for eps in gk["smc"]["eps"]])
    ax.axhline(np.median(final_eps), color=pal["ref"],
               linestyle="--", linewidth=1.0,
               label=fr"$\epsilon_T$ médian = {np.median(final_eps):.3g}")
    ax.set_yscale("log")
    ax.set_xlabel("étape SMC $t$")
    ax.set_ylabel(r"$\epsilon_t$")
    ax.legend(frameon=False)
    _save(fig, "fig_eps_schedule_GK", bw)


# ==========================================================================
#  FIG 9 -- Coût vs erreur (NN), illustre la domination Pareto
# ==========================================================================
@_both
def fig_cost_vs_error_NN(nn, bw=False):
    pal = _palette(bw)
    fig, ax = plt.subplots(figsize=(6, 4))

    for key, label, marker in (("gibbs", "ABC-Gibbs", "o"),
                               ("smc",   "SMC-ABC",   "s")):
        cost = nn[key]["n_sims"]
        err = nn[key]["mean_wass_mu"]
        ax.errorbar(cost.mean(), err.mean(),
                    xerr=cost.std(ddof=1), yerr=err.std(ddof=1),
                    marker=marker, markersize=8,
                    color=pal[key], ecolor=pal[key],
                    linestyle="none", capsize=4,
                    label=label, markerfacecolor="none" if bw else pal[key],
                    markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nombre de simulations modèle (log)")
    ax.set_ylabel(r"$\overline{W_1}(\mu_j)$ (log)")
    ax.legend(frameon=False)
    _save(fig, "fig_cost_vs_error_NN", bw)


# ==========================================================================
#                           POINT D'ENTRÉE
# ==========================================================================
def make_all(nn, gk):
    setup_style()
    fig_boxplot_W1_NN(nn)
    fig_posterior_alpha_NN(nn)
    fig_posterior_mu_NN(nn)
    fig_traceplots_NN(nn)
    fig_eps_schedule_NN(nn)
    fig_boxplot_predictive_GK(gk)
    fig_posterior_alpha_GK(gk)
    fig_eps_schedule_GK(gk)
    fig_cost_vs_error_NN(nn)
