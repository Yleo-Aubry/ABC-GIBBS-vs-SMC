"""
============================================================================
  make_all_figures.py
  --------------------------------------------------------------------------
  Régénère toutes les figures du rapport Projet 4 d'un seul coup, à
  partir d'un seed fixé. Utilise un cache pickle (`cache/*.pkl`) pour
  éviter de relancer les simulations si elles existent déjà.

  Usage :
      python make_all_figures.py              # utilise le cache si présent
      python make_all_figures.py --force      # relance toutes les sims
      python make_all_figures.py --R 10       # nb de réplicats
============================================================================
"""
from __future__ import annotations

import argparse
import numpy as np

from data_collection import collect_all
import figures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--R", type=int, default=10,
                   help="nombre de réplicats indépendants (défaut 10)")
    p.add_argument("--seed", type=int, default=100,
                   help="base seed (défaut 100)")
    p.add_argument("--force", action="store_true",
                   help="relance les sims même si le cache existe")
    args = p.parse_args()

    np.random.seed(args.seed)        # cohérence du seed numpy global
    print(f"[1/2] collecte des sorties (R={args.R}, seed={args.seed},"
          f" force={args.force})")
    nn, gk = collect_all(R=args.R, base_seed=args.seed, force=args.force)

    print("[2/2] génération des figures (color + bw) -> figures/")
    figures.make_all(nn, gk)
    print("[ok] terminé.")


if __name__ == "__main__":
    main()
