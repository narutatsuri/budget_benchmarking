#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulates label-collection experiments under a fixed budget to compare:
- The probability of correctly ranking two classifiers (a better one and a worse one),
- The observed accuracy gap vs. the true gap.

Command-line arguments allow specification of parameter sweeps and output locations.
"""
import os
import argparse
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

def simulate_experiment(k, m, q, p, eps, random_state=None):
    """
    Simulate a single experiment trial for given parameters.

    Parameters
    ----------
    k : int
        Total labeling budget.
    m : int
        Number of labels (annotators) per instance.
    q : float
        Annotator accuracy (probability of matching the true label).
    p : float
        Accuracy of the better classifier.
    eps : float
        Accuracy gap between the better and worse classifier (worse = p - eps).
    random_state : np.random.RandomState, optional
        Random state for reproducibility. If None, uses np.random.

    Returns
    -------
    dict
        A dictionary containing:
        - "rank_better": int (1 if the better classifier is ranked higher, else 0)
        - "observed_gap": float (the difference in observed accuracies of the two classifiers,
          each measured against the aggregated test label).
    """
    if random_state is None:
        random_state = np.random

    # Number of distinct data points
    n = k // m

    # Better vs. worse classifier accuracy
    p_b = p
    p_w = p - eps

    # Generate n true labels
    true_labels = random_state.randint(0, 2, size=n)

    # Generate m labels per data point
    # annotator_draws[i,j] < q => annotator j is correct on data point i
    annotator_draws = random_state.rand(n, m)
    annot_labels = np.zeros((n, m), dtype=int)

    for i in range(n):
        for j in range(m):
            is_correct = (annotator_draws[i, j] < q)
            if is_correct:
                annot_labels[i, j] = true_labels[i]
            else:
                annot_labels[i, j] = 1 - true_labels[i]

    # Aggregate via majority vote
    majority_label = (annot_labels.sum(axis=1) > (m / 2)).astype(int)

    # Simulate classifier correctness w.r.t. true labels
    # Then compare classifier predictions to majority_label
    classifier_b_draws = random_state.rand(n)
    classifier_b_pred = np.where(classifier_b_draws < p_b,
                                 true_labels, 1 - true_labels)

    classifier_w_draws = random_state.rand(n)
    classifier_w_pred = np.where(classifier_w_draws < p_w,
                                 true_labels, 1 - true_labels)

    # Calculate observed accuracies against aggregated test label
    acc_b = np.mean(classifier_b_pred == majority_label)
    acc_w = np.mean(classifier_w_pred == majority_label)

    observed_gap = acc_b - acc_w
    rank_better = 1 if (acc_b > acc_w) else 0

    return {
        "rank_better": rank_better,
        "observed_gap": observed_gap
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ks", type=int, nargs="+", default=[100, 500, 1000, 5000],
                        help="List of total labeling budgets.")
    parser.add_argument("--Ms", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100],
                        help="List of number of labels per instance (m).")
    parser.add_argument("--Qs", type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.9],
                        help="List of annotator accuracies.")
    parser.add_argument("--Ps", type=float, nargs="+", default=[0.6, 0.7, 0.75, 0.8, 0.9],
                        help="List of better classifier accuracies.")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Base increment for gap values.")
    parser.add_argument("--simulations", type=int, default=1000,
                        help="Number of simulation trials per parameter configuration.")
    parser.add_argument("--save-dir", type=str, default="results/",
                        help="Directory to save output CSV.")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Optional name for output CSV file.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    random_state = np.random.RandomState(args.seed)

    rows = []
    # Collect all parameter combinations
    param_combos = list(product(args.Ks, args.Ms, args.Qs, args.Ps))

    for k, m, q, p in tqdm(param_combos, ncols=80):
        # For each classifier accuracy p, define a set of epsilons up to p*1.0, e.g.
        run_eps = [args.epsilon * i for i in range(1, int(p * 10))]
        # Filter out any eps >= p
        run_eps = [e for e in run_eps if e < p]

        for eps in run_eps:
            rank_better_vals = []
            observed_gap_vals = []

            for _ in range(args.simulations):
                res = simulate_experiment(k, m, q, p, eps, random_state=random_state)
                rank_better_vals.append(res["rank_better"])
                observed_gap_vals.append(res["observed_gap"])

            observed_gap_arr = np.array(observed_gap_vals)
            rank_success = np.mean(rank_better_vals)
            gap_mean = np.mean(observed_gap_arr)
            gap_std = np.std(observed_gap_arr, ddof=1)
            gap_rmse = np.sqrt(np.mean((observed_gap_arr - eps) ** 2))

            row = {
                "k": k,
                "m": m,
                "q": q,
                "p": p,
                "eps": eps,
                "rank_success": rank_success,
                "gap_mean": gap_mean,
                "gap_std": gap_std,
                "gap_rmse": gap_rmse,
            }
            rows.append(row)

    df_results = pd.DataFrame(rows)

    if args.save_name is not None:
        out_path = os.path.join(args.save_dir, args.save_name)
    else:
        out_path = os.path.join(args.save_dir, "results.csv")

    df_results.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
