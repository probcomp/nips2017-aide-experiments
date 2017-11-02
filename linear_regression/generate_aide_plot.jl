using DataFrames
using Query
using PyPlot
using LaTeXStrings

include("schema.jl")

const STDERR_ALPHA = 0.3
const PLOT_DIR="plots"
const DATA_DIR = "data"
const WIDTH = 3
const HEIGHT = 3
const SHARED_YLABEL = "AIDE estimate (nats)"
const SHARED_YLIM = [-0.25, 9]
const TITLE_FONTSIZE=12

function add_baseline()
    ax = plt[:gca]()
    ax[:set_xscale]("log")
    xlim = ax[:get_xlim]()
    plt[:plot](xlim, [0, 0], color="black", alpha=0.5)
    ax[:set_xlim](xlim)
end

function plot_smc_data(data::DataFrame)
    num_rejuvenation = 2
    for k in unique(data[COL_NUM_METAINFERENCE])
        subdata  = @from i in data begin
            @where i.num_metainference == k
            @select {i.num_particles, 
                     i.aide_estimate,
                     i.aide_estimate_standard_error}
            @collect DataFrame
        end
        exponent = Int(log10(k))
        num_particles = subdata[COL_NUM_PARTICLES]
        est = subdata[COL_AIDE_ESTIMATE]
        err = subdata[COL_AIDE_STDERR]
        label=latexstring("\$M_t = 10^$exponent\$")
        plt[:plot](num_particles, est, label=label)
        plt[:fill_between](num_particles, est-err, est+err, alpha=STDERR_ALPHA)
    end
    add_baseline()
    plt[:gca]()[:set_ylim](SHARED_YLIM)
    plt[:xlabel]("Number of particles")
    plt[:ylabel](SHARED_YLABEL)
    plt[:legend]()
    plt[:title]("Sequential Monte Carlo", fontsize=TITLE_FONTSIZE)
end


function plot_bbvi_data(data::DataFrame)
    est = data[COL_AIDE_ESTIMATE]
    err = data[COL_AIDE_STDERR]
    num_vi_iters = data[COL_NUM_SGD_ITERS]
    plt[:plot](num_vi_iters, est, label="Variational")
    kl_asymptote = est[end] # assumes sorted.
    plt[:fill_between](num_vi_iters, est - err, est + err, alpha=STDERR_ALPHA)
    add_baseline()
    xlim = [minimum(num_vi_iters), maximum(num_vi_iters)]
    plt[:plot](xlim, [kl_asymptote, kl_asymptote], color="black", alpha=0.5)
    plt[:gca]()[:set_xlim](xlim)
    plt[:gca]()[:set_ylim](SHARED_YLIM)
    plt[:xlabel]("Number of gradient steps")
    plt[:ylabel](SHARED_YLABEL)
    plt[:title]("Variational Inference", fontsize=TITLE_FONTSIZE)
end

function plot_mcmc_data(data::DataFrame)
    for k in unique(data[COL_NUM_METAINFERENCE])
        subdata  = @from i in data begin
            @where i.num_metainference == k
            @select {i.num_mcmc_iters,
                     i.aide_estimate,
                     i.aide_estimate_standard_error}
            @collect DataFrame
        end
        exponent = Int(log10(k))
        num_mcmc_iters = subdata[COL_NUM_MCMC_ITERS]
        est = subdata[COL_AIDE_ESTIMATE]
        err = subdata[COL_AIDE_STDERR]
        label=latexstring("\$M_t = 10^$exponent\$")
        plt[:plot](1 + num_mcmc_iters, est, label=label)
        plt[:fill_between](1 + num_mcmc_iters, est-err, est+err, alpha=STDERR_ALPHA)
    end
    add_baseline()
    plt[:gca]()[:set_ylim](SHARED_YLIM)
    plt[:xlabel]("1 + Number of transitions")
    plt[:ylabel](SHARED_YLABEL)
    plt[:title]("Metropolis-Hastings", fontsize=TITLE_FONTSIZE)
end

println("plotting divergence bounds..")

plt[:figure](figsize=(WIDTH*3,HEIGHT))

println("plotting SMC data...")
smc_data = readtable("$DATA_DIR/smc_data.csv")
plt[:subplot](1, 3, 1)
plot_smc_data(smc_data)

println("plotting MCMC data...")
mcmc_data = readtable("$DATA_DIR/mcmc_data.csv")
plt[:subplot](1, 3, 2)
plot_mcmc_data(mcmc_data)

println("plotting BBVI data...")
bbvi_data = readtable("$DATA_DIR/bbvi_data.csv")
plt[:subplot](1, 3, 3)
plot_bbvi_data(bbvi_data)

println("Saving output plot.")
plt[:tight_layout]()
plt[:savefig]("$PLOT_DIR/linear_regression_combined.pdf")

println("done!")
