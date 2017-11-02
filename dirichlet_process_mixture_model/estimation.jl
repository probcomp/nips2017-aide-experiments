include("schema.jl")

function estimate_aide(gold_standard_sampler::Sampler, target_sampler::Sampler, num_replicates::Int)
    aide_estimates = aide(gold_standard_sampler, target_sampler, 1, 1, num_replicates, num_replicates)
    estimate = mean(aide_estimates)
    stderr = std(aide_estimates) / sqrt(num_replicates)
    return (estimate, stderr)
end

function estimate_expected_num_clusters(target_sampler::Sampler, num_replicates::Int)
    target_samples = pmap((i) -> simulate(target_sampler)[1], 1:num_replicates)
    num_clusters = map((s) -> length(clusters(s.crp)), target_samples)
    num_clusters = convert(Vector{Float64}, num_clusters)
    estimate = mean(num_clusters)
    stderr = std(num_clusters) / sqrt(length(num_clusters))
    return (estimate, stderr)
end

function generate_sampler_data(data::Vector{Float64}, hypers::Hyperparameters,
                               num_aide_replicates::Int, num_diagnostic_replicates::Int,
                               gold_standard_sampler::Sampler)
    proposal_name_column = String[]
    num_sweeps_column = Int[]
    num_particles_column = Int[]
    aide_estimate_column = Float64[]
    aide_stderr_column = Float64[]
    num_clusters_estimate_column = Float64[]
    num_clusters_stderr_column = Float64[]

    for num_sweeps in num_sweeps_list
        for num_particles in num_particles_list
            prior_target_sampler = make_prior_proposal_smc_sampler(data, hypers, num_sweeps, num_particles)
            optimal_target_sampler = make_optimal_proposal_smc_sampler(data, hypers, num_sweeps, num_particles)

            target_samplers = [(prior_target_sampler, PRIOR_PROPOSAL_NAME),
                               (optimal_target_sampler, OPTIMAL_PROPOSAL_NAME)]

            for (target_sampler, proposal_name) in target_samplers
                println("$proposal_name, num_sweeps=$num_sweeps, num_particles=$num_particles")
                push!(proposal_name_column, proposal_name)
                push!(num_sweeps_column, num_sweeps)
                push!(num_particles_column, num_particles)

                # AIDE estimate
                # NOTE: it is possible to reuse the gold-standard samplers when evaluating different target samplers
                # We do not do this here.
                (aide_estimate, aide_stderr) = estimate_aide(gold_standard_sampler, target_sampler, num_aide_replicates)
                push!(aide_estimate_column, aide_estimate)
                push!(aide_stderr_column, aide_stderr)

                # estimate the expected number of clusters
                (num_clusters_estimate, num_clusters_stderr) = estimate_expected_num_clusters(target_sampler, num_diagnostic_replicates)
                push!(num_clusters_estimate_column, num_clusters_estimate)
                push!(num_clusters_stderr_column, num_clusters_stderr)
            end
        end
    end

    data = DataFrame()
    data[COL_PROPOSAL_NAME] = proposal_name_column
    data[COL_NUM_PARTICLES] = num_particles_column
    data[COL_NUM_SWEEPS] = num_sweeps_column 
    data[COL_AIDE_ESTIMATE] = aide_estimate_column
    data[COL_AIDE_STDERR] = aide_stderr_column
    data[COL_NUM_CLUSTERS_ESTIMATE] = num_clusters_estimate_column
    data[COL_NUM_CLUSTERS_STDERR] = num_clusters_stderr_column
    return data
end

