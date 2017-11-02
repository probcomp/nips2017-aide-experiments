To generate data (takes 24 minutes using 62 processes each with a 1.4 GHz CPU):
```
julia -p <num-procs> generate_data.jl
```
This generates CSV files in `data/`.

To generate plot:
```
julia generate_aide_plot.jl
```
This generates a PDF file:
```
linear_regression_combined.pdf
```
