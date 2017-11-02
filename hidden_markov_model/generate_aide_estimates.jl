import SMC
import Distributions
using DataFrames
include("hmm_serialization.jl")
include("schema.jl")

@everywhere begin
    using SMC
    using Distributions
    include("smc_samplers.jl")
    include("../aide.jl")
end

function generate_aide_estimates(hmm::HiddenMarkovModel,
                                 observations::Vector{Int},
                                 num_particles_list::Vector{Int},
                                 num_metainference_list::Vector{Int},
                                 num_replicates::Int)
    data = DataFrame()
    num_particles_column = Int[]
    num_metainference_column = Int[]
    proposal_name_column = String[]
    aide_estimate_column = Float64[]
    aide_stderr_column = Float64[]
    gold_standard_name_column = String[]

    exact_sampler = HMMExactSampler(hmm, observations)
    gold_standard_sampler = make_optimal_proposal_smc_sampler(hmm, observations, 1000)
    
    for num_particles in num_particles_list
        for num_metainference in num_metainference_list
            println("generating data for num_partices=$num_particles, num_metainference=$num_metainference...")
            
            prior_smc_sampler = make_prior_proposal_smc_sampler(hmm, observations, num_particles)
            kls = aide(exact_sampler, prior_smc_sampler, 1, num_metainference, num_replicates, num_replicates)
            push!(num_particles_column, num_particles)
            push!(num_metainference_column, num_metainference)
            push!(proposal_name_column, PRIOR_PROPOSAL_NAME)
            push!(aide_estimate_column, mean(kls))
            push!(aide_stderr_column, std(kls)/sqrt(length(kls)))
            push!(gold_standard_name_column, EXACT_GOLD_STANDARD)

            prior_smc_sampler = make_prior_proposal_smc_sampler(hmm, observations, num_particles)
            kls = aide(gold_standard_sampler, prior_smc_sampler, 1, num_metainference, num_replicates, num_replicates)
            push!(num_particles_column, num_particles)
            push!(num_metainference_column, num_metainference)
            push!(proposal_name_column, PRIOR_PROPOSAL_NAME)
            push!(aide_estimate_column, mean(kls))
            push!(aide_stderr_column, std(kls)/sqrt(length(kls)))
            push!(gold_standard_name_column, APPROXIMATE_GOLD_STANDARD)

            optimal_smc_sampler = make_optimal_proposal_smc_sampler(hmm, observations, num_particles)
            kls = aide(exact_sampler, optimal_smc_sampler, 1, num_metainference, num_replicates, num_replicates)
            push!(num_particles_column, num_particles)
            push!(num_metainference_column, num_metainference)
            push!(proposal_name_column, OPTIMAL_PROPOSAL_NAME)
            push!(aide_estimate_column, mean(kls))
            push!(aide_stderr_column, std(kls)/sqrt(length(kls)))
            push!(gold_standard_name_column, EXACT_GOLD_STANDARD)

            optimal_smc_sampler = make_optimal_proposal_smc_sampler(hmm, observations, num_particles)
            kls = aide(gold_standard_sampler, optimal_smc_sampler, 1, num_metainference, num_replicates, num_replicates)
            push!(num_particles_column, num_particles)
            push!(num_metainference_column, num_metainference)
            push!(proposal_name_column, OPTIMAL_PROPOSAL_NAME)
            push!(aide_estimate_column, mean(kls))
            push!(aide_stderr_column, std(kls)/sqrt(length(kls)))
            push!(gold_standard_name_column, APPROXIMATE_GOLD_STANDARD)
        end
    end
    data[COL_NUM_PARTICLES] = num_particles_column
    data[COL_NUM_METAINFERENCE] = num_metainference_column
    data[COL_PROPOSAL_NAME] = proposal_name_column
    data[COL_AIDE_ESTIMATE] = aide_estimate_column
    data[COL_AIDE_STDERR] = aide_stderr_column
    data[COL_GOLD_STANDARD_NAME] = gold_standard_name_column
    return data
end

# do experiment
data_dir = "data"
plot_dir = "plots"

# load HMM and observations
hmm = load_hmm("$data_dir/hmm.json")
num_states = num_states(hmm)
num_obs = num_observations(hmm)
observations = load_observations("$data_dir/observations.json")
num_steps = length(observations)

# do AIDE experiment, save data to CSV file
num_particles_list = [1, 3, 10, 30, 100]
num_metainference_list = [1, 100]
num_replicates = 100
aide_estimates = generate_aide_estimates(hmm, observations, num_particles_list,
                                         num_metainference_list, num_replicates)
writetable("$data_dir/aide_estimates.csv", aide_estimates)
println("done!")
