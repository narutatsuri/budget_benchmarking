## Budget Benchmarking

Code and visualizations for performing parameter sweep to compare multi-label versus single-label strategies in classifier benchmarking. We are interested in evaluating ranking correctness and estimating the performance gap between classifiers under varying budget and noise conditions.


### Parameter Sweep

The parameter sweep for both ranking correctness and margin estimation can be run with the following command:

```bash
python parameter-sweep.py --Ks [list of budgets] \
                          --Ms [list of annotations per data point] \
                          --Ps [list of better classifier accuracies] \
                          --Qs [list of annotator accuracies] \
                          --eps [epsilon step size] \
                          --simulations [number of simulations per configuration] \
                          --seed [random seed]
```

When no parameters are passed, the default parameter sweeps are run:
 - `Ks`: 100, 500, 1000, 5000
 - `Ms`: 1, 2, 5, 10, 25, 50
 - `Qs`: 0.6, 0.7, 0.75, 0.8, 0.9
 - `Ps`: 0.6, 0.7, 0.75, 0.8, 0.9
 - `seed`: 42
 - `epsilon`: 0.1
 - `simulations`: 1000

The script iterates over the specified parameter combinations, runs the designated number of simulations for each, and aggregates results into a CSV file saved in the specified directory.


### Visualizations

Visualization scripts are available in the Jupyter notebook `notebooks/plot_results.ipynb`. Pre-generated visualizations are stored in the `results/plots/` directory:
 - Files prefixed with `gap-` pertain to performance gap estimation results.
 - Files prefixed with `rank-` pertain to ranking correctness results.
Additional summary heatmaps and the complete results dataset can be found in the `results/ `directory. The full parameter sweep results are provided in `results/results_full.csv`.