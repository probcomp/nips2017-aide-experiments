using SMC
abstract type Sampler end

function estimate_elbo(p::Sampler, q::Sampler, KP::Int, KQ::Int)
    p_log_weights = Array{Float64,1}(KP)
    q_log_weights = Array{Float64,1}(KQ)
    x, p_log_weights[1] = simulate(p)
    for k=2:KP
        p_log_weights[k] = regenerate(p, x)
    end
    for k=1:KQ
        q_log_weights[k] = regenerate(q, x)
    end
    ((logsumexp(q_log_weights) - log(KQ)) -
     (logsumexp(p_log_weights) - log(KP)))
end

function estimate_elbo_par(p::Sampler, q::Sampler, KP::Int, KQ::Int, N::Int)
    f = (i::Int) -> estimate_elbo(p, q, KP, KQ)
    estimates::Array{Float64,1} = pmap(f, 1:N)
    estimates
end

function aide(gold_standard_sampler::Sampler, target_sampler::Sampler,
              num_gold_standard_metainference::Int, num_target_metainference::Int,
              num_gold_standard_inference::Int, num_target_inference::Int)
    a = -estimate_elbo_par(gold_standard_sampler, target_sampler,
                           num_gold_standard_metainference, num_target_metainference,
                           num_gold_standard_inference)
    b = -estimate_elbo_par(target_sampler, gold_standard_sampler,
                           num_target_metainference, num_gold_standard_metainference,
                           num_target_inference)
    return a + b
end

immutable SMCSampler <: Sampler
    scheme::SMCScheme
    # the evalutaor for the target density p(x,y)
    log_joint_probability_evaluator
end

function simulate(p::SMCSampler)
    x, lml_estimate = smc(p.scheme)
    log_weight = p.log_joint_probability_evaluator(x) - lml_estimate
    (x, log_weight)
end

function regenerate(p::SMCSampler, x)
    lml_estimate = conditional_smc(p.scheme, x)
    p.log_joint_probability_evaluator(x) - lml_estimate
end

immutable HMMExactSampler <: Sampler
    hmm::HiddenMarkovModel
    observations::Array{Int,1}
end

function simulate(p::HMMExactSampler)
    x = hmm_posterior_sample(p.hmm, p.observations)
    log_weight = hmm_log_joint_probability(p.hmm, x, p.observations)
    (x, log_weight)
end

function regenerate(p::HMMExactSampler, x)
    hmm_log_joint_probability(p.hmm, x, p.observations)
end
