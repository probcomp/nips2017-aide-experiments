using DataFrames
import SMC


@everywhere struct Problem
    x::Vector{Float64}
    y::Vector{Float64}
    prior_std::Float64
    noise_std::Float64
end

@everywhere begin
    include("../aide.jl")
    include("exact_closed_form.jl")
    include("smc.jl")
    include("bbvi.jl")
    include("mcmc.jl")
end

include("schema.jl")

function time_sampler(sampler::Sampler)
    const NUM_TIMING_RUNS = 100
    times = []
    for i=1:NUM_TIMING_RUNS
        tic()
        simulate(sampler)
        push!(times, toq())
    end
    median(times)
end

function generate_prior_data(problem::Problem, num_replicates::Int)
    println("generating prior data...")
    exact_sampler = ExactLinregSampler(problem)
    prior_sampler = PriorSampler(problem.x, problem.y, problem.prior_std, problem.noise_std)
    kls = aide(exact_sampler, prior_sampler, 1, 1, num_replicates, num_replicates)

    data = DataFrame()
    data[COL_SAMPLER_NAME] = ["prior"]
    data[COL_AIDE_ESTIMATE] = [mean(kls)]
    data[COL_AIDE_STDERR] = [std(kls) / sqrt(num_replicates)]
    data[COL_MEDIAN_RUNTIME] = [time_sampler(prior_sampler)]
    return data
end

function generate_posterior_data(problem::Problem, num_replicates::Int)
    println("generating posterior data...")
    exact_sampler = ExactLinregSampler(problem)
    kls = aide(exact_sampler, exact_sampler, 1, 1, num_replicates, num_replicates)

    data = DataFrame()
    data[COL_SAMPLER_NAME] = ["posterior"]
    data[COL_AIDE_ESTIMATE] = [mean(kls)]
    data[COL_AIDE_STDERR] = [std(kls) / sqrt(num_replicates)]
    data[COL_MEDIAN_RUNTIME] = [time_sampler(exact_sampler)]
    return data
end


function generate_smc_data(problem::Problem, num_replicates::Int)
    println("generating SMC data...")
    exact_sampler = ExactLinregSampler(problem)

    k_column = Int[]
    num_rejuvenation_column = Int[]
    num_particles_column = Int[]
    aide_est_column = Float64[]
    aide_stderr_column = Float64[]
    median_runtime_column = Float64[]

    all_ks = [1, 10, 10^3]
    all_num_rejuvenation = [2] #[0, 1, 2, 4, 8] # TODO add back to make the rejuv. plot for appendix
    all_num_particles = [1, 2, 3, 6, 10, 20, 30, 60, 100]
    for k in all_ks
        for num_rejuvenation in all_num_rejuvenation
            for num_particles in all_num_particles
                println("k=$k, num_rejuvenation=$num_rejuvenation, num_particles=$num_particles")
                push!(k_column, k)
                push!(num_rejuvenation_column, num_rejuvenation)
                push!(num_particles_column, num_particles)
                smc = LinregSMCSampler(problem.x, problem.y, problem.prior_std, problem.noise_std, num_rejuvenation, num_particles)

                kls = aide(exact_sampler, smc, 1, k, num_replicates, num_replicates)
                push!(aide_est_column, mean(kls))
                push!(aide_stderr_column, std(kls) / sqrt(num_replicates))
                push!(median_runtime_column, time_sampler(smc))
            end
        end
    end

    data = DataFrame()
    data[COL_SAMPLER_NAME] = fill("smc", length(aide_est_column))
    data[COL_NUM_METAINFERENCE] = k_column
    data[COL_NUM_REJUVENATION] = num_rejuvenation_column
    data[COL_NUM_PARTICLES] = num_particles_column
    data[COL_AIDE_ESTIMATE] = aide_est_column
    data[COL_AIDE_STDERR] = aide_stderr_column
    data[COL_MEDIAN_RUNTIME] = median_runtime_column
    return data
end

function generate_bbvi_data(problem::Problem, num_replicates::Int)
    println("generating BBVI data...")
    exact_sampler = ExactLinregSampler(problem)

    all_num_vi_iters = convert(Array{Int,1},floor.(10.^[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]))
    bbvi_num_samples = 10
    bbvi_a = 10000.0
    bbvi_b = 0.75

    num_sgd_iters_column = Int[]
    aide_est_column = Float64[]
    aide_stderr_column = Float64[]
    median_runtime_column = Float64[]
    for num_iters in all_num_vi_iters
        println("num_iters=$num_iters")
        push!(num_sgd_iters_column, num_iters)
        params = BBVIParams(0.9, bbvi_num_samples, bbvi_a, bbvi_b, num_iters, problem.prior_std, problem.noise_std) # 1000 # was 0.7
        vi_sampler = BBVISampler(params, problem.x, problem.y)
        kls = aide(exact_sampler, vi_sampler, 1, 1, num_replicates, num_replicates)
        push!(aide_est_column, mean(kls))
        push!(aide_stderr_column, std(kls) / sqrt(num_replicates))
        times = []
        for i=1:num_replicates
            tic()
            simulate(vi_sampler)
            push!(times, toq())
        end
        push!(median_runtime_column, median(times))
    end

    data = DataFrame()
    data[COL_SAMPLER_NAME] = fill("smc", length(aide_est_column))
    data[COL_NUM_METAINFERENCE] = fill(1, length(aide_est_column))
    data[COL_NUM_SGD_ITERS] = num_sgd_iters_column 
    data[COL_AIDE_ESTIMATE] = aide_est_column
    data[COL_AIDE_STDERR] = aide_stderr_column
    data[COL_MEDIAN_RUNTIME] = median_runtime_column
    return data
end


function generate_mcmc_data(problem::Problem, num_replicates::Int)
    println("generating MCMC data...")
    exact_sampler = ExactLinregSampler(problem)

    k_column = Int[]
    num_mcmc_iters_column = Int[]
    aide_est_column = Float64[]
    aide_stderr_column = Float64[]
    median_runtime_column = Float64[]

    all_num_mh_iters = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    all_ks = [1, 10, 10^3]
    for num_iters in all_num_mh_iters
        mh_sampler = LinregMHSampler(problem.x, problem.y, problem.prior_std, problem.noise_std, num_iters)
        for k in all_ks
            println("num_iters=$num_iters, k=$k")
            push!(k_column, k)
            push!(num_mcmc_iters_column, num_iters)
            kls = aide(exact_sampler, mh_sampler, 1, k, num_replicates, num_replicates)
            push!(aide_est_column, mean(kls))
            push!(aide_stderr_column, std(kls) / sqrt(num_replicates))
            push!(median_runtime_column, time_sampler(mh_sampler))
        end
    end

    data = DataFrame()
    data[COL_SAMPLER_NAME] = fill("mcmc", length(aide_est_column))
    data[COL_NUM_METAINFERENCE] = k_column
    data[COL_NUM_MCMC_ITERS] = num_mcmc_iters_column 
    data[COL_AIDE_ESTIMATE] = aide_est_column
    data[COL_AIDE_STDERR] = aide_stderr_column
    data[COL_MEDIAN_RUNTIME] = median_runtime_column
    return data
end

problem = Problem([-2.0, 2.0], [-2.0, 2.0], 1.0, 1.0)
num_replicates =  10000
data_dir = "data"
if !Base.Filesystem.isdir(data_dir)
    Base.Filesystem.mkdir(data_dir)
end

prior_data = generate_prior_data(problem, num_replicates)
writetable("$data_dir/prior_data.csv", prior_data)

posterior_data = generate_posterior_data(problem, num_replicates)
writetable("$data_dir/posterior_data.csv", posterior_data)

smc_data = generate_smc_data(problem, num_replicates)
writetable("$data_dir/smc_data.csv", smc_data)

bbvi_data = generate_bbvi_data(problem, num_replicates)
writetable("$data_dir/bbvi_data.csv", bbvi_data)

mcmc_data = generate_mcmc_data(problem, num_replicates)
writetable("$data_dir/mcmc_data.csv", mcmc_data)
