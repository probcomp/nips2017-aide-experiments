using SMC
using Distributions

type LinregState
    # these change
    intercept::Float64
    slope::Float64
    
    # these don't change they are parameters
    prior_std::Float64
    noise_std::Float64
end

# JUST USE INDEPENDENT MH PROPOSAL FOR REJUVENATION

immutable Datum
    x::Float64
    y::Float64
end

function log_likelihood(datum::Datum, state::LinregState)
    pred_y = state.intercept + state.slope * datum.x
    dist = Normal(pred_y, state.noise_std)
    logpdf(dist, datum.y)
end

immutable LinregInitializer
    datum::Datum
    prior_std::Float64
    noise_std::Float64
end

function SMC.forward(init::LinregInitializer)
    # sample the intercept and slope from the prior, weight by the likeliood
    intercept = rand(Normal(0.0, init.prior_std))
    slope = rand(Normal(0.0, init.noise_std))
    state = LinregState(intercept, slope, init.prior_std, init.noise_std)
    log_weight = log_likelihood(init.datum, state)
    (state, log_weight)
end

function SMC.backward(init::LinregInitializer, state::LinregState)
    log_likelihood(init.datum, state)
end

immutable LinregIncrementer
    data::Array{Datum,1} # this is the full list of all data, the incrementer only uses a prefix of it
    datum_index::Int
    rejuvenation_iters::Int
    function LinregIncrementer(data::Array{Datum,1}, datum_index::Int,
                               rejuvenation_iters::Int)
        @assert datum_index > 1
        new(data, datum_index, rejuvenation_iters)
    end
end

function independent_mh_step!(state::LinregState, data::Array{Datum,1})
    # propose from the prior
    intercept_old = state.intercept
    slope_old = state.slope
    old_ll = 0.0
    for datum in data
        old_ll += log_likelihood(datum, state)
    end
    state.intercept = rand(Normal(0.0, state.prior_std))
    state.slope = rand(Normal(0.0, state.prior_std))
    new_ll = 0.0
    for datum in data
        new_ll += log_likelihood(datum, state)
    end
    if log(rand()) >= new_ll - old_ll
        # reject
        state.intercept = intercept_old
        state.slope = slope_old
    end
end

function SMC.forward(incr::LinregIncrementer, state::LinregState)
    new_state = deepcopy(state)
    # do rejuvenation over already incorporate data points
    incorporated_data = incr.data[1:incr.datum_index-1]
    for i=1:incr.rejuvenation_iters
        independent_mh_step!(new_state, incorporated_data)
    end
    # incorporate new datapoint and compute log likelihood as weight
    log_weight = log_likelihood(incr.data[incr.datum_index], new_state)
    (new_state, log_weight)
end

function SMC.backward(incr::LinregIncrementer, new_state::LinregState)
    # compute log likelihood for weight, before rejuvenation
    log_weight = log_likelihood(incr.data[incr.datum_index], new_state)

    state = deepcopy(new_state)
    # do rejuvenation sweep over data
    incorporated_data = incr.data[1:incr.datum_index-1]
    for i=1:incr.rejuvenation_iters
        independent_mh_step!(state, incorporated_data)
    end
    (state, log_weight)
end

function LinregSMCScheme(x::Array{Float64,1}, y::Array{Float64,1}, prior_std::Float64, noise_std::Float64, 
                         rejuvenation_iters::Int, num_particles::Int)
    @assert length(x) == length(y)
    data = Array{Datum,1}()
    for (xi, yi) in zip(x, y)
        push!(data, Datum(xi, yi))
    end
    initializer = LinregInitializer(data[1], prior_std, noise_std)
    incrementers = Array{LinregIncrementer,1}(length(data)-1)
    for i=2:length(data)
        incrementers[i-1] = LinregIncrementer(data, i, rejuvenation_iters)
    end
    SMCScheme(initializer, incrementers, num_particles)
end

function log_joint_density(state::LinregState, x::Array{Float64,1}, y::Array{Float64,1})
    log_prior = logpdf(Normal(0., state.prior_std), state.intercept)
    log_prior += logpdf(Normal(0., state.prior_std), state.slope)
    log_likelihood = 0.0
    for (xi, yi) in zip(x, y)
        y_expected = state.intercept + state.slope * xi
        log_likelihood += logpdf(Normal(y_expected, state.noise_std), yi)
    end
    log_prior + log_likelihood
end

immutable LinregSMCSampler <: Sampler
    inner_module::SMCSampler
    prior_std::Float64
    noise_std::Float64
    function LinregSMCSampler(x::Array{Float64,1}, y::Array{Float64,1}, prior_std::Float64, noise_std::Float64,
                             rejuvenation_iters::Int, num_particles::Int)
        smc_scheme = LinregSMCScheme(x, y, prior_std, noise_std, rejuvenation_iters, num_particles)
        inner_module = SMCSampler(smc_scheme, (sample) -> log_joint_density(sample, x, y))
        new(inner_module, prior_std, noise_std)
    end
end

function simulate(sampler::LinregSMCSampler)
    (sample, log_weight) = simulate(sampler.inner_module)
    ([sample.intercept, sample.slope], log_weight)
end

function regenerate(sampler::LinregSMCSampler, intercept_and_slope::Array{Float64,1})
    @assert length(intercept_and_slope) == 2
    (intercept, slope) = intercept_and_slope
    state = LinregState(intercept, slope, sampler.prior_std, sampler.noise_std)
    regenerate(sampler.inner_module, state)
end
