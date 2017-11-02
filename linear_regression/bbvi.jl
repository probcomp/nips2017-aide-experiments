using Distributions

function sample_q(intercept_mu::Float64, slope_mu::Float64, std::Float64, num::Int)
    intercepts = rand(Normal(intercept_mu, std), num)
    slopes = rand(Normal(slope_mu, std), num)
    (intercepts, slopes)
end

function log_q(intercept_mu::Float64, slope_mu::Float64, intercept::Float64, slope::Float64, std::Float64)
    intercept_dist = Normal(intercept_mu, std)
    slope_dist = Normal(slope_mu, std)
    logpdf(intercept_dist, intercept) + logpdf(slope_dist, slope)
end

function log_q_grad(intercept_mu::Float64, slope_mu::Float64, intercept::Float64, slope::Float64, std::Float64)
    intercept_grad = (intercept - intercept_mu) / (std * std)
    slope_grad = (slope - slope_mu) / (std * std)
    [intercept_grad, slope_grad]
end

function log_likelihood(x::Array{Float64,1}, y::Array{Float64,1}, noise_std::Float64, intercept::Float64, slope::Float64)
    line = x * slope + intercept
    ll = 0.0
    for (y_i, line_i) in zip(y, line)
        dist = Normal(line_i, noise_std)
        ll += logpdf(dist, y_i)
    end
    ll
end

function log_p_unnormalized(problem::Problem, intercept::Float64, slope::Float64)
    intercept_dist = Normal(0.0, problem.prior_std)
    slope_dist = Normal(0.0, problem.prior_std)
    log_prior = logpdf(intercept_dist, intercept)
    log_prior += logpdf(slope_dist, slope)
    log_lik = log_likelihood(problem.x, problem.y, problem.noise_std, intercept, slope)
    log_prior + log_lik
end

function gradient_est(params::Array{Float64,1}, problem::Problem, std::Float64, num::Int)
    @assert length(params) == 2
    intercept_mu = params[1]
    slope_mu = params[2]
    (intercepts, slopes) = sample_q(intercept_mu, slope_mu, std, num)
    elbo_est = 0.0
    grad_est = zeros(2)
    for (intercept, slope) in zip(intercepts, slopes)
        lj = log_p_unnormalized(problem, intercept, slope)
        lq = log_q(intercept_mu, slope_mu, intercept, slope, std)
        diff = lj - lq
        elbo_est += diff
        grad_est += diff * log_q_grad(intercept_mu, slope_mu, intercept, slope, std)
    end
    elbo_est /= num
    grad_est /= num
    (grad_est, elbo_est)
end

immutable BBVIResult
    intercept_mu::Float64
    slope_mu::Float64
    elbo_est::Float64
end

function linreg_fast_bbvi_foreign(x::Array{Float64,1}, y::Array{Float64,1}, prior_std::Float64, noise_std::Float64, iters::Int, std::Float64, num::Int, a::Float64, b::Float64)
    # num is the number of samples from q to use to approximate the gradient
    # (a + t)^(-b) is the step size at time t
    params = zeros(2)
    elbo_est = NaN
    problem = Problem(x, y, prior_std, noise_std)
    @assert iters > 0
    start = time() # in seconds
    for iter in 1:iters
        rho = (a + iter) ^ (-b)
        (grad_est, elbo_est) = gradient_est(params, problem, std, num)
        params += rho * grad_est
    end
    intercept_mu = params[1]
    slope_mu = params[2]
    BBVIResult(intercept_mu, slope_mu, elbo_est)
end

immutable BBVIParams
    # std of the intercept and slope normals of the variational family (constant)
    std::Float64

    # Number of samples to use for gradient estimates
    num::Int

    # Robbins Munro parameters
    a::Float64
    b::Float64

    # iterations of gradient ascent (no early stopping)
    iters::Int

    # model parameters
    prior_std::Float64
    noise_std::Float64
end

immutable BBVISampler <: Sampler
    intercept_dist::Normal
    slope_dist::Normal

    # when you construct the BBVI it does the optimization
    function BBVISampler(params::BBVIParams, x::Array{Float64,1}, y::Array{Float64,1})
        result = linreg_fast_bbvi_foreign(x, y, params.prior_std, params.noise_std, 
                                          params.iters, params.std, params.num, params.a, params.b)

        intercept_dist = Normal(result.intercept_mu, params.std)
        slope_dist = Normal(result.slope_mu, params.std)
        new(intercept_dist, slope_dist)
    end
end

function simulate(sampler::BBVISampler)
    intercept = rand(sampler.intercept_dist)
    slope = rand(sampler.slope_dist)
    sample = [intercept, slope]
    log_weight = regenerate(sampler, sample)
    (sample, log_weight)
end

function regenerate(sampler::BBVISampler, intercept_and_slope::Array{Float64,1})
    @assert length(intercept_and_slope) == 2
    (intercept, slope) = intercept_and_slope
    logpdf(sampler.intercept_dist, intercept) + logpdf(sampler.slope_dist, slope)
end





