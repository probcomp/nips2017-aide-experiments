using Distributions

# sampling from exact partial posterior
function linreg_exact_posterior(x::Array{Float64,1}, y::Array{Float64,1}, prior_std::Float64, noise_std::Float64)
    # intercept comes first, then slope
    n = length(x)
    @assert n == length(y)
    S0 = prior_std * prior_std * eye(2)
    phi = hcat(ones(n), x)
    @assert size(phi) == (n, 2)
    noise_var = noise_std * noise_std
    noise_precision = 1./noise_var
    SN = inv(inv(S0) + noise_precision * (phi' * phi))
    mN = SN * ((inv(S0) * zeros(2)) + noise_precision * (phi' * y))
    # return the mean vector and covariance matrix
    (mN, SN)
end

immutable ExactLinregSampler <: Sampler
    dist::MvNormal
    function ExactLinregSampler(prior_std::Float64, noise_std::Float64, x::Array{Float64,1}, y::Array{Float64,1})
        (mN, SN) = linreg_exact_posterior(x, y, prior_std, noise_std)
        dist = MvNormal(mN, SN)
        new(dist)
    end
end

function ExactLinregSampler(problem::Problem)
    ExactLinregSampler(problem.prior_std, problem.noise_std, problem.x, problem.y)
end

function simulate(sampler::ExactLinregSampler)
    intercept_and_slope = rand(sampler.dist)
    log_weight = logpdf(sampler.dist, intercept_and_slope)
    (intercept_and_slope, log_weight)
end

function regenerate(sampler::ExactLinregSampler, intercept_and_slope::Array{Float64,1})
    @assert length(intercept_and_slope) == 2
    logpdf(sampler.dist, intercept_and_slope)
end
