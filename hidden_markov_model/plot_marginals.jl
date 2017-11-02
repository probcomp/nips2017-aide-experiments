using SMC
using PyPlot
include("../aide.jl")
include("smc_samplers.jl")
include("hmm_serialization.jl")

function add_axis_labels()
    fontsize=14
    ax = plt[:gca]()
    ax[:set_ylabel]("state", labelpad=-15, fontsize=fontsize)
    ax[:set_xlabel]("time", labelpad=-15, fontsize=fontsize)
end

function format_axes()
    add_axis_labels()
    plt[:tight_layout](pad=0)
end

function add_upper_right_label(label::String)
    plt[:text](0.86, 0.69, label, fontsize=34, transform=plt[:gcf]()[:transFigure])
end

function render_marginals(hmm::HiddenMarkovModel, observations::Vector{Int},
                          sampler::Sampler, num_replicates::Int, label="")
    num_steps = length(observations)
    marginals = zeros(hmm.num_states, num_steps)
    for i=1:num_replicates
        states, _ = simulate(sampler)
        for (t, state) in enumerate(states)
            marginals[state, t] += 1.0
        end
    end
    marginals = marginals / num_replicates
    render_hmm_posterior_marginals!(hmm, marginals, false, [1, 50], [1, 20])
    add_upper_right_label(label)
    format_axes()
end

# where to read data from
DATA_DIR = "data"

# where to write plots to
PLOT_DIR = "plots"
if !Base.Filesystem.isdir(PLOT_DIR)
    Base.Filesystem.mkdir(PLOT_DIR)
end


# load HMM, ground truth states, and observations
hmm = load_hmm("$DATA_DIR/hmm.json")
num_states = num_states(hmm)
num_obs = num_observations(hmm)
ground_truth_states = load_states("$DATA_DIR/states.json")
observations = load_observations("$DATA_DIR/observations.json")
num_steps = length(ground_truth_states)
if length(observations) != num_steps
    error("Dimension mismatch")
end

# fix random seed for Monte Carlo estimates of marginals
srand(1)

const NUM_MARGINAL_REPLICATES = 1000

# plot the ground truth states
println("Plotting ground truth states...")
width = 3
height = 1.1*width*num_states/num_steps
plt[:figure](figsize=(width,height))
render_hmm_states!(hmm, ground_truth_states, false, [1, 50], [1, 20])
format_axes()
plt[:savefig]("$PLOT_DIR/ground_truth_states.pdf")

# exact posterior marginals
println("Plotting exact posterior marginals...")
exact_sampler = HMMExactSampler(hmm, observations)
plt[:figure](figsize=(width,height))
render_marginals(hmm, observations, exact_sampler, NUM_MARGINAL_REPLICATES)
plt[:savefig]("$PLOT_DIR/exact_posterior_marginals.pdf")

# prior proposal, 1 particle
println("Plotting prior proposal, 1 particle, marginals...")
sampler = make_prior_proposal_smc_sampler(hmm, observations, 1)
plt[:figure](figsize=(width,height))
render_marginals(hmm, observations, sampler, NUM_MARGINAL_REPLICATES, "A")
plt[:savefig]("$PLOT_DIR/smc_prior_1p_marginals.pdf")

# prior proposal, 10 particles
println("Plotting prior proposal, 10 particles, marginals...")
sampler = make_prior_proposal_smc_sampler(hmm, observations, 10)
plt[:figure](figsize=(width,height))
render_marginals(hmm, observations, sampler, NUM_MARGINAL_REPLICATES, "B")
plt[:savefig]("$PLOT_DIR/smc_prior_10p_marginals.pdf")

# optimal proposal, 100 particles
println("Plotting optimal proposal, 100 particles, marginals...")
sampler = make_optimal_proposal_smc_sampler(hmm, observations, 100)
plt[:figure](figsize=(width,height))
render_marginals(hmm, observations, sampler, NUM_MARGINAL_REPLICATES, "C")
plt[:savefig]("$PLOT_DIR/smc_cond_100p_marginals.pdf")

# optimal proposal, 1000 particles
println("Plotting optimal proposal, 1000 particles, marginals...")
sampler = make_optimal_proposal_smc_sampler(hmm, observations, 1000)
plt[:figure](figsize=(width,height))
render_marginals(hmm, observations, sampler, NUM_MARGINAL_REPLICATES)
plt[:savefig]("$PLOT_DIR/smc_cond_1000p_marginals.pdf")

println("done!")
