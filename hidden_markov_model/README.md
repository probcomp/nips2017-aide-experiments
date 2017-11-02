To generate AIDE estimates (takes 2-3 minutes with 62 processes with 1 CPU each at 1.4 GHz each)
```
julia -p <num-procs> generate_aide_estimates.jl
```

This produces an output CSV containing a table of AIDE estimates for different conditions:
```
data/aide_estimates.csv
```

To generate AIDE estimate plots:
```
julia plot_aide.jl
```

To generate the marginals plots (takes 1-2 minutes with 2.6 GHz CPU)
```
julia plot_marginals.jl
```
