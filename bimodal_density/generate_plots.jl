using Distributions
using SMC
using KernelDensity
using PyPlot
using LaTeXStrings

include("../math.jl")
include("../aide.jl")

# where to write plots to
const PLOT_DIR = "plots"
if !Base.Filesystem.isdir(PLOT_DIR)
    Base.Filesystem.mkdir(PLOT_DIR)
end

"""
An unnormalized target density defined by a Gaussian prior and a mixture-of-Gaussian likelihood.
"""
struct Problem
    prior_mean::Float64
    prior_std::Float64
    proportions::Array{Float64,1}
    likelihood_means::Array{Float64,1}
    likelihood_stds::Array{Float64,1}
end

function log_joint_density(problem::Problem, x::Float64)
    prior = Normal(problem.prior_mean, problem.prior_std)
    ld = logpdf(prior, x)
    log_likelihoods = Array{Float64,1}(length(problem.likelihood_means))
    for i=1:length(problem.likelihood_means)
        likelihood = Normal(problem.likelihood_means[i], 
                            problem.likelihood_stds[i])
        log_likelihoods[i] = log(problem.proportions[i]) + logpdf(likelihood, x)
    end
    ld + logsumexp(log_likelihoods)
end

"""
Exact sampler for the given inference problem.
"""
struct ExactSampler <: Sampler
    problem::Problem
end

function simulate(s::ExactSampler)
    error("not implemented!")
end

function regenerate(s::ExactSampler, x)
    log_joint_density(s.problem, x)
end

"""
Sampling importance resampling scheme.

The distribution is a Gaussian distribution with mean `mean` and
standard-deviation `std`.  `SMC.forward` Samples from the proposal and returns
the importance weight with respect to the given inference problem `p`.
`SMC.backward` returns the importance weight given a latent sample.
"""
struct GaussianInitializer
    p::Problem
    mean::Float64
    std::Float64
end

function SMC.forward(init::GaussianInitializer)
    dist = Normal(init.mean, init.std)
    x = rand(dist)
    (x, log_joint_density(init.p, x) - logpdf(dist, x))
end

function SMC.backward(init::GaussianInitializer, x::Float64)
    dist = Normal(init.mean, init.std)
    log_joint_density(init.p, x) - logpdf(dist, x)
end

function make_sir_scheme(problem::Problem,
                   init_mean::Float64,
                   init_std::Float64,
                   num_particles::Int)
    initializer = GaussianInitializer(problem, init_mean, init_std)
    SMCScheme(initializer, [], num_particles)
end

function make_offset_proposal_scheme(problem::Problem, num_particles::Int)
    make_sir_scheme(problem, 2.0, 0.5, num_particles)
end

function make_broad_proposal_scheme(problem::Problem, num_particles::Int)
    make_sir_scheme(problem, 0.0, 1.0, num_particles)
end

function make_sampler(sir_scheme::SMCScheme, problem::Problem)
    SMCSampler(sir_scheme, (x) -> log_joint_density(problem, x))
end

const OFFSET_PROPOSAL = "offset"
const BROAD_PROPOSAL = "broad"

"""
Returns a dictionary mapping key to a vector of tuples (sample, log marginal
likelihood estimate)
"""
function run_sir(problem::Problem, num_particles_list::Vector{Int}, num_replicates::Int)
    sir_outputs = Dict()
    for n in num_particles_list 
        key = (OFFSET_PROPOSAL, n)
        scheme = make_offset_proposal_scheme(problem, n)
        sir_outputs[key] = map((i) -> smc(scheme), 1:num_replicates)
        key = (BROAD_PROPOSAL, n)
        scheme = make_broad_proposal_scheme(problem, n)
        sir_outputs[key] = map((i) -> smc(scheme), 1:num_replicates)
    end
    return sir_outputs
end

function get_log_marginal_likelihood_estimates(sir_outputs::Dict, proposal_name::String, num_particles::Int)
    map((output) -> output[2],  sir_outputs[(proposal_name, num_particles)])
end

function get_samples(sir_outputs::Dict, proposal_name::String, num_particles::Int)
    map((output) -> output[1],  sir_outputs[(proposal_name, num_particles)])
end


"""
Returns a dictionary mapping key to kernel density estimators
"""
function kernel_density_estimators(sir_outputs::Dict)
    kd_estimators = Dict()
    for (key, output) in sir_outputs
        samples = map((o) -> o[1], output)
        kd_estimators[key] = InterpKDE(kde(samples))
    end
    return kd_estimators
end

function plot_sir_sampling_densities(problem::Problem, kd_estimators::Dict, fname::String)

    width = 2.5
    height = 2.5
    plt[:figure](figsize=(width * 1.3, height))
    ax = plt[:gca]()

    # number of particles to plot sampling densities for
    all_num_particles_to_plot_samples = [1, 10, 100, 1000]

    # location of sampling densities on the y-axis
    yticks = [4, 3, 2, 1]

    # location of floor of plot on the y-axis
    density_y_base = 0

    # height of the target density on the y-axis
    density_actual_span = 0.7
    
    # x-axis limits
    data_lim = (-3, 4)

    # x-coordinates at which to evaluate densities
    density_xs = linspace(-3, 4, 100)

    # plot the unnormalized target density
    density_ys = map((x) -> exp(log_joint_density(problem, x)), density_xs)
    density_ys_span = maximum(density_ys) - minimum(density_ys)
    density_ys *= density_actual_span / density_ys_span
    plt[:plot](density_xs, density_y_base + density_ys, color="black")

    # plot the sampling densities (kernel density estimates)
    for (i, num_particles) in enumerate(all_num_particles_to_plot_samples)
        ys = pdf(kd_estimators[(OFFSET_PROPOSAL, num_particles)], density_xs)
        ys = (ys / maximum(ys)) * density_actual_span + yticks[i]
        plt[:plot](density_xs, ys, color="magenta", zorder=100, alpha=0.7)
        ys = pdf(kd_estimators[(BROAD_PROPOSAL, num_particles)], density_xs)
        ys = (ys / maximum(ys)) * density_actual_span + yticks[i]
        plt[:plot](density_xs, ys, color="blue", alpha=0.7)
    end

    # labels
    plt[:text](1.5, density_y_base + density_actual_span/9.0, "posterior\ndensity", fontsize=10)
    plt[:text](-1.7, density_y_base + density_actual_span/6.0, "L", fontsize=12)
    plt[:text](0.3, density_y_base + density_actual_span/6.0, "R", fontsize=12)
    ax[:set_xlim](data_lim)
    exponents = [0, 1, 2, 3]
    exponent_strings = map((i) -> @sprintf("%d", i), exponents)
    ax[:set_yticks](yticks)
    ax[:set_yticklabels](map((exponent_string) -> latexstring("\$10^{$exponent_string}\$"),
                             exponent_strings))
    ax[:tick_params](axis="both", which="major", labelsize=9)
    plt[:ylabel]("Number of particles\n")

    # arrow for the number of particles on y-axis
    ax[:annotate]("", xy=(-0.2, 0.0), xycoords="axes fraction", xytext=(-0.2, 1.0), 
                arrowprops=Dict([("arrowstyle", "->"), ("color", "black")]))

    plt[:tight_layout]()
    plt[:savefig](fname)
end

function draw_arrow(x, y, xlen, ylen, ax)
    ax[:arrow](x, y, xlen, ylen, fc="black", ec="black",
	    transform=ax[:transAxes], length_includes_head=true,
	    head_width=0.05, head_length=0.05)
end

function plot_log_marginal_likelihoods(sir_outputs::Dict, all_num_particles::Vector{Int}, fname::String)

    function plot_single_series(proposal_name::String, label, color)
        # a vector of a vector of IID estimates
        # there is one vector of IID estimates for each number of particles
        individual_estimates = map((num_particles) -> get_log_marginal_likelihood_estimates(sir_outputs, proposal_name, num_particles), all_num_particles)
        estimates = map(mean, individual_estimates)
        stderrs = map((data::Vector{Float64}) -> std(data)/sqrt(length(data)), individual_estimates)
        plt[:plot](all_num_particles, estimates, color=color, label=label)
        lower = estimates - stderrs
        upper = estimates + stderrs
        plt[:fill_between](all_num_particles, lower, estimates, color=color, alpha=0.3)
        plt[:fill_between](all_num_particles, estimates, upper, color=color, alpha=0.3)
    end

    width = 2.5
    height = 2.5
    plt[:figure](figsize=(width, height))
    ax = plt[:gca]()

    # plot log marginal likelihood estimates
    plot_single_series(OFFSET_PROPOSAL, "Offset proposal", "magenta")
    plot_single_series(BROAD_PROPOSAL, "Broad proposal", "blue")
    gold_standard_lml = mean(get_log_marginal_likelihood_estimates(sir_outputs, "broad", maximum(all_num_particles)))
    xlim = ax[:get_xlim]()
    plt[:plot](xlim, gold_standard_lml * ones(2), color="orange", label="Gold-standard")

    # arrow and label for small penalty
    plt[:text](12, -26, "Small penalty")
    draw_arrow(0.8, 0.73, 0.0, 0.19, ax)

    # plot formatting and labels
    ax[:set_xscale]("log")
    plt[:minorticks_off]()
    ax[:tick_params](axis="both", which="major", labelsize=8)
    ax[:set_xlim](xlim)
    plt[:legend](loc="lower right", fontsize=8.7)
    plt[:xlabel]("Number of particles")
    plt[:title]("Log marginal\nlikelihood (nats)")
    plt[:tight_layout]()
    plt[:savefig](fname)
end

function aide_estimate(gold_standard_sampler::Sampler, target_sampler::Sampler, num_replicates::Int)
    lowers = map((i::Int) -> estimate_elbo(target_sampler, gold_standard_sampler, 1, 1),
                 1:num_replicates)
    uppers = map((i::Int) -> -estimate_elbo(gold_standard_sampler, target_sampler, 1, 1),
                 1:num_replicates)
    return (uppers - lowers)
end

function generate_aide_estimates(problem::Problem, all_num_particles::Vector{Int}, num_replicates::Int)
    gold_standard_scheme = make_broad_proposal_scheme(problem, maximum(all_num_particles))
    gold_standard_sampler = make_sampler(gold_standard_scheme, problem)
    aide_estimates = Dict()
    for num_particles in all_num_particles
        offset_proposal_scheme = make_offset_proposal_scheme(problem, num_particles)
        offset_proposal_sampler = make_sampler(offset_proposal_scheme, problem)
        broad_proposal_scheme = make_broad_proposal_scheme(problem, num_particles)
        broad_proposal_sampler = make_sampler(broad_proposal_scheme, problem)
        aide_estimates[(OFFSET_PROPOSAL, num_particles)] = aide_estimate(gold_standard_sampler, offset_proposal_sampler, num_replicates)
        aide_estimates[(BROAD_PROPOSAL, num_particles)] = aide_estimate(gold_standard_sampler, broad_proposal_sampler, num_replicates)
    end
    return aide_estimates
end

function plot_aide_estimates(aide_estimates::Dict, all_num_particles::Vector{Int}, fname::String)

    function plot_single_series(proposal_name::String, label, color)
        # a vector of a vector of IID estimates
        # there is one vector of IID estimates for each number of particles
        individual_estimates = map((num_particles) -> aide_estimates[(proposal_name, num_particles)],
                                   all_num_particles)
        estimates = map(mean, individual_estimates)
        stderrs = map((data::Vector{Float64}) -> std(data)/sqrt(length(data)), individual_estimates)
        plt[:plot](all_num_particles, estimates, color=color, label=label)
        lower = estimates - stderrs
        upper = estimates + stderrs
        plt[:fill_between](all_num_particles, lower, estimates, color=color, alpha=0.3)
        plt[:fill_between](all_num_particles, estimates, upper, color=color, alpha=0.3)
    end

    width = 2.5
    height = 2.5
    plt[:figure](figsize=(width, height))
    ax = plt[:gca]()

    # plot log marginal likelihood estimates
    plot_single_series(OFFSET_PROPOSAL, "Offset proposal", "magenta")
    plot_single_series(BROAD_PROPOSAL, "Broad proposal", "blue")

    # arrow and label for large penalty
    plt[:text](10, 29, "Large penalty for\nmissing mode")
    draw_arrow(0.8, 0.36, 0.0, -0.22, ax)

    # plot formatting and labels
    ymax = ax[:get_ylim]()[2]
    ax[:set_ylim](0, ymax)
    ax[:set_xscale]("log")
    plt[:minorticks_off]()
    ax[:tick_params](axis="both", which="major", labelsize=8)
    plt[:legend](loc="upper right", fontsize=8.7)
    plt[:xlabel]("Number of particles")
    plt[:title]("AIDE estimate\n(nats)")
    plt[:tight_layout]()
    plt[:savefig](fname)
end


# generate plots
srand(1)
problem = Problem(0.0, 10.0, [0.7, 0.3], [-1, 1], [0.1, 0.1])
exponents = linspace(0, 3, 10)
num_particles_list = Vector{Int}(floor.(10.^exponents))

# generate subfigure (a)
println("Generating density plot...")
sir_outputs_for_kde = run_sir(problem, num_particles_list, 10000)
kd_estimators = kernel_density_estimators(sir_outputs_for_kde)
plot_sir_sampling_densities(problem, kd_estimators, "$PLOT_DIR/bimodal_densities.pdf")

# generate subfigure (b)
println("Generating AIDE estimate plot...")
aide_estimates = generate_aide_estimates(problem, num_particles_list, 100)
plot_aide_estimates(aide_estimates, num_particles_list, "$PLOT_DIR/bimodal_symkl.pdf")

# generate subfigure (c)
println("Generating log marginal likelihood plot...")
sir_outputs_for_lml = run_sir(problem, num_particles_list, 100)
plot_log_marginal_likelihoods(sir_outputs_for_lml, num_particles_list, "$PLOT_DIR/bimodal_lml.pdf")
