import SMC
using PyPlot 
import Distributions
import JSON
using DataFrames

@everywhere begin
    using SMC
    using Distributions
    include("dpmm.jl")
    include("smc_samplers.jl")
    include("../aide.jl")
end

include("estimation.jl")

# where to load and save data
const DATA_DIR = "data"

# load dataset
data = JSON.parsefile("$DATA_DIR/num_clusters_comparison_data.json")
data = convert(Vector{Float64}, data)
num_data = length(data)

# set prior
m_prior = Normal(0, 1) # mean cluster center 
r_prior = Gamma(1, 1) # r * data-precision is the precision of cluster centers
nu_prior = Gamma(1, 1) # expected data-precision of cluster is nu/s
s_prior = Gamma(1, 1)
nign_params_prior = NIGNParamsPrior(m_prior, r_prior, nu_prior, s_prior)
alpha_prior = Gamma(1, 1)
hypers = Hyperparameters(nign_params_prior, alpha_prior)

num_sweeps_list = [0, 4]
num_particles_list= [1, 2, 4, 8, 10, 20, 40, 80, 100]

# the number of runs of inference to use for diagnostic is twice the number of
# AIDE replicates, because each AIDE replicate involves running the target
# inference sampler twice
num_aide_replicates = 62
num_diagnostic_replicates = num_aide_replicates * 2
println("generating sampler data...")
gold_standard_sampler = make_optimal_proposal_smc_sampler(data, hypers, 4, 100)
data = generate_sampler_data(data, hypers, num_aide_replicates, num_diagnostic_replicates,
                             gold_standard_sampler)
writetable("$DATA_DIR/sampler_data.csv", data)
println("done!")
