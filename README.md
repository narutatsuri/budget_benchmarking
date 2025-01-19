## Running the Code
### Parameter Sweep
```
python parameter-sweep.py --Ks  [list of integers for budget] \ 
                          --Ms  [list of annotations per data point (n is determined automatically)] \ 
                          --Ps  [list of better classifier's accuracy] \ 
                          --Qs  [list of annotator accuracy] \ 
                          --eps [float for epsilon step size] \ 
                          --Ms  [seed] \ 
                          --T   [number of simulations to run per configuration]
```
When no parameters are passed, the default parameter sweeps are run:
 - `Ks`: 100, 1000, 5000
 - `Ms`: 1, 2, 5, 20, 50
 - `Qs`: 0.6, 0.7, 0.8, 0.9
 - `Ps`: 0.6, 0.7, 0.8, 0.9
 - `seed`: 42
 - `epsilon`: 0.1
 - `simulations`: 1000

### Visualizations
Code for visualizations are in `notebooks/plot_results.ipynb`.