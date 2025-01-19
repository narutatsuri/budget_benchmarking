import os
import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm


def simulate_experiment(k, m, q, p, eps, random_state=None):
    """
    Simulate single run for fixed (k, m, q, p, eps).

    Args:
        k (int): total budget
        m (int): number of labels per instance
        q (float): accuracy (prob of correct label) of each annotator
        p (float): accuracy of the better classifier
        eps (float): gap between better (p) and worse (p-eps)
        random_state (np.random.RandomState): optional for reproducibility

    Returns:
        dict with:
            - 'rank_better': 1 if the better classifier is ranked higher, 0 otherwise
            - 'observed_gap': observed (estimated) gap in accuracy = (acc_b - acc_w)
              where each acc is measured against the test label.
    """
    if random_state is None:
        random_state = np.random

    n = k // m

    p_b = p
    p_w = p - eps

    # Generate n true labels
    true_labels = random_state.randint(0, 2, size=n)

    # Generate m labels each correct w.p. q, then majority vote for final test label
    annotator_draws = random_state.rand(n, m)

    # For each data point i, each annotator j is correct if annotator_draws[i,j] < q
    annot_labels = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            is_correct = annotator_draws[i, j] < q
            if is_correct:
                # correct label
                annot_labels[i, j] = true_labels[i]
            else:
                # flipped label
                annot_labels[i, j] = 1 - true_labels[i]

    # Aggregate if m>1
    majority_label = (annot_labels.sum(axis=1) > (m / 2)).astype(int)

    # Evaluate better & worse classifier predictions vs test label
    classifier_b_draws = random_state.rand(n)
    classifier_b_pred = np.where(classifier_b_draws < p_b, true_labels, 1 - true_labels)

    classifier_w_draws = random_state.rand(n)
    classifier_w_pred = np.where(classifier_w_draws < p_w, true_labels, 1 - true_labels)

    # Measure accuracy vs. aggregated test label
    acc_b = np.mean(classifier_b_pred == majority_label)
    acc_w = np.mean(classifier_w_pred == majority_label)

    observed_gap = acc_b - acc_w
    rank_better = 1 if (acc_b > acc_w) else 0

    return {"rank_better": rank_better, "observed_gap": observed_gap}


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("--Ks",          type=int,   nargs="+", default=[100, 500, 1000, 5000])
parser.add_argument("--Ms",          type=int,   nargs="+", default=[1, 5, 10, 25, 50, 100])
parser.add_argument("--Qs",          type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.9])
parser.add_argument("--Ps",          type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.9])
parser.add_argument("--seed",        type=int,              default=42)
parser.add_argument("--epsilon",     type=float,            default=0.1)
parser.add_argument("--simulations", type=int,              default=1000)
parser.add_argument("--save-dir",    type=str,              default="results/")
parser.add_argument("--save-name",   type=str,              default=None)
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

random_state = np.random.RandomState(args.seed)

rows = []

for k, m, q, p in tqdm(
    [comb for comb in product(args.Ks, args.Ms, args.Qs, args.Ps)], ncols=80
):
    run_eps = [eps for eps in [args.epsilon * i for i in range(1, int(p * 10))]]

    for eps in run_eps:

        rank_better = []
        observed_gap = []

        for _ in range(args.simulations):
            res = simulate_experiment(k, m, q, p, eps, random_state=random_state)

            rank_better.append(res["rank_better"])
            observed_gap.append(res["observed_gap"])

        observed_gap = np.array(observed_gap)

        row = {
            "k":            k,
            "m":            m,
            "q":            q,
            "p":            p,
            "eps":          eps,
            "rank_success": np.mean(np.array(rank_better)),
            "gap_mean":     np.mean(observed_gap),
            "gap_std":      np.std(observed_gap, ddof=1),
            "gap_rmse":     np.sqrt(np.mean((observed_gap - eps) ** 2)),
        }
        rows.append(row)

pd.DataFrame(rows).to_csv(
    (
        os.path.join(args.save_dir, args.save_name)
        if args.save_name is not None
        else os.path.join(args.save_dir, "results.csv")
    ),
    index=False,
)
