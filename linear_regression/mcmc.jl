immutable State
    intercept::Float64
    slope::Float64
end

immutable Datum
    x::Float64
    y::Float64
end

immutable LinregMHSampler <: Sampler
    num_steps::Int
    prior_std::Float64
    noise_std::Float64
    data::Array{Datum,1}
    function LinregMHSampler(x::Array{Float64,1}, y::Array{Float64,1}, 
                            prior_std::Float64, noise_std::Float64, num_steps::Int)
        @assert length(x) == length(y)
        data = Array{Datum,1}()
        for (xi, yi) in zip(x, y)
            push!(data, Datum(xi, yi))
        end
        new(num_steps, prior_std, noise_std, data)
    end
end

function prior_sample(sampler::LinregMHSampler)
    intercept = rand(Normal(0., sampler.prior_std))
    slope = rand(Normal(0., sampler.prior_std))
    State(intercept, slope)
end

function log_prior(sampler::LinregMHSampler, state::State)
    lp = logpdf(Normal(0., sampler.prior_std), state.intercept)
    lp += logpdf(Normal(0., sampler.prior_std), state.slope)
    lp
end

function log_likelihood(sampler::LinregMHSampler, state::State)
    ll = 0.0
    for datum in sampler.data
        y_expected = state.intercept + state.slope * datum.x
        ll += logpdf(Normal(y_expected, sampler.noise_std), datum.y)
    end
    ll  
end

function log_unnormalized_posterior(sampler::LinregMHSampler, state::State)
    log_prior(sampler, state) + log_likelihood(sampler, state)
end

function imh_step(sampler::LinregMHSampler, state::State)
    # perform one step of independent MH
    # propose from prior
    state_proposed = prior_sample(sampler)
    if log(rand()) <= log_likelihood(sampler, state_proposed) - log_likelihood(sampler, state)
        # accept
        new_state = state_proposed
    else
        # reject
        new_state = state
    end
    new_state
end

function run_markov_chain(sampler::LinregMHSampler, state::State, steps::Int)
    for step in 1:steps
        state = imh_step(sampler, state)
    end
    state
end

function run_inference(sampler::LinregMHSampler)
    init_state = prior_sample(sampler)
    state = run_markov_chain(sampler, init_state, sampler.num_steps)
    # returns the initial and final state of the Markov chain
    # the initial state is all we need from the trace
    return (init_state, state)
end

function log_weight(sampler::LinregMHSampler, init_state::State, output::State)
    lw = 0.
    lw += log_prior(sampler, init_state) 
    lw += log_unnormalized_posterior(sampler, output)
    lw -= log_unnormalized_posterior(sampler, init_state)
    lw
end

function simulate(sampler::LinregMHSampler)
    (init_state, state) = run_inference(sampler)
    ([state.intercept, state.slope], log_weight(sampler, init_state, state))
end

function regenerate(sampler::LinregMHSampler, intercept_and_slope::Array{Float64,1})
    @assert length(intercept_and_slope) == 2
    intercept, slope = intercept_and_slope
    state = State(intercept, slope)
    init_state = run_markov_chain(sampler, state, sampler.num_steps)
    log_weight(sampler, init_state, state)
end


immutable PriorSampler <: Sampler
    x::Array{Float64,1}
    y::Array{Float64,1}
    prior_std::Float64
    noise_std::Float64
end

function simulate(sampler::PriorSampler)
    dist = Normal(0., sampler.prior_std)
    intercept = rand(dist)
    slope = rand(dist)
    ([intercept, slope], logpdf(dist, intercept) + logpdf(dist, slope))
end

function regenerate(sampler::PriorSampler, intercept_and_slope::Array{Float64,1})
    (intercept, slope) = intercept_and_slope
    dist = Normal(0., sampler.prior_std)
    logpdf(dist, intercept) + logpdf(dist, slope)
end








