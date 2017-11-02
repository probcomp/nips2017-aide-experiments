Code for experiments and figure-generation in:

> Cusumano-Towner, Mansinghka. AIDE: An algorithm for measuring the accuracy of probabilistic inference algorithms. To appear in Proceedings of Advances in Neural Information Processing Systems (NIPS), 2017.

Each of the figures in the paper is represented by a directory.
Directories contain code for runing experiments, the data from which figures were generated, and the code for generating figures.
Note that for the Monte Carlo estimates for which parallelism was used, the data from which figures were generated is not reproducible (although it is provided).

## Dependencies

Tested with Julia 0.6.0.

Depends on the following publicly registered registered Julia packages:

 - DataFrames                    0.10.1
 - Distributions                 0.15.0
 - KernelDensity                 0.3.2
 - Query                         0.7.2
 - PyPlot                        2.3.2
 - LaTeXStrings                  0.3.0

Also depends on the SMC.jl Julia package, which is not publicly registered.
To install SMC.jl use:

```
julia> Pkg.clone("git@github.com:probcomp/SMC.jl.git")
```
