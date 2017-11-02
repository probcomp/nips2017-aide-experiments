To run tests:
```
julia runtests.jl
```

To generate data for Figure 5 (takes 17 minutes with 62 worker processes each with an allocated 1.4GHz CPU):
```
julia -p <num-processes> generate_data.jl
```

To generate data for the galaxy data set experiment in the supplement (takes 401 minutes with 62 worker processes each with an allocated 1.4GHz CPU):
```
julia -p <num-processes> galaxies.jl
```

NOTE: The majority of the runtime is due to execution of the gold-standard sampler or its meta-inference algorithm.
AIDE allows executions of the gold-standard sampler to be reused across evaluations of different target inference algorithms.
Therefore, estimating the KL divergence from the sampling distribution of the gold-standard to the sampling distribution of a set of target algorithms should be substantially faster than this experiment.
However, estimating the KL divergence from the sampling distribution of a target distribution to the sampling distribution of the gold-standard algorithm requires running the gold-standard meta-inference sampler with the target algorithm sample provided as input.
Therefore, estimating the KL divergence from the target to the gold standard (or the symmetric divergence) requires execution of gold-standard meta-inference independently for each target inference algorithm.

To generate plots for Figure 5:
```
julia generate_plots.jl
```
This produces three PDFs:
```
plots/aide.pdf
plots/num_clusters.pdf
plots/legend.pdf
```

To generate plots for the galaxies experiment in the supplement:
```
julia galaxies_plot.jl
```
This produces four PDFs:
```
plots/galaxies_histogram.pdf
plots/galaxies_aide.pdf
plots/galaxies_num_clusters.pdf
plots/galaxies_legend.pdf
```
