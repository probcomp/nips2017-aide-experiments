using SMC

function make_prior_proposal_smc_sampler(hmm::HiddenMarkovModel,    
                                         observations::Vector{Int}, 
                                         num_particles::Int)
    function log_joint_evaluate(latent_sample::Vector{Int})
        return hmm_log_joint_probability(hmm, latent_sample, observations)
    end
    smc_scheme = HMMPriorSMCScheme(hmm, observations, num_particles)
    return SMCSampler(smc_scheme, log_joint_evaluate)
end

function make_optimal_proposal_smc_sampler(hmm::HiddenMarkovModel,    
                                           observations::Vector{Int}, 
                                           num_particles::Int)
    function log_joint_evaluate(latent_sample::Vector{Int})
        return hmm_log_joint_probability(hmm, latent_sample, observations)
    end
    smc_scheme = HMMConditionalSMCScheme(hmm, observations, num_particles)
    return SMCSampler(smc_scheme, log_joint_evaluate)
end
