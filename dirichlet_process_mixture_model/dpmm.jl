using Distributions
using SMC
using PyPlot
using LaTeXStrings
import DataStructures.OrderedDict

include("../math.jl")
include("crp.jl")
include("nign.jl")

struct Hyperparameters
    nign_params_prior::NIGNParamsPrior
    alpha_prior::Gamma
end

type DPMMState
    alpha::Float64
    crp::CRPState # contains the assignments
    components::OrderedDict{Int, NIGN}
    nign_params::NIGNParams
    hypers::Hyperparameters
end

function cluster_likelihoods(state::DPMMState)
    map((component) -> log_probability_density(component, state.nign_params),
        values(state.components))
end

function log_joint_density(state::DPMMState)
    ld = logpdf(state.hypers.alpha_prior, state.alpha)
    ld += log_probability_density(state.hypers.nign_params_prior, state.nign_params)
    ld += log_probability(state.crp, state.alpha)
    ld += sum(cluster_likelihoods(state))
    ld
end

function conditional(state::DPMMState, i::Int, datum::Float64)
    @assert !has_assignment_for(state.crp, i)
    next = next_new_cluster(state.crp)
    state.components[next] = NIGN()
	logp = Vector{Float64}(length(state.components))
    clusters = collect(keys(state.components))
    crp_logpdf_before = log_probability(state.crp, state.alpha)
    components_logpdf_before = cluster_likelihoods(state)
	for (j, cluster) in enumerate(clusters)
        @assert cluster == next_new_cluster(state.crp) || has_cluster(state.crp, cluster)
        component = state.components[cluster]
		incorporate!(component, datum)
        incorporate!(state.crp, i, cluster)
        logp[j] = log_probability(state.crp, state.alpha) - crp_logpdf_before
		logp[j] += log_probability_density(component, state.nign_params) - components_logpdf_before[j]
		unincorporate!(component, datum)
        unincorporate!(state.crp, i)
        @assert next == next_new_cluster(state.crp)
	end
    log_denom = logsumexp(logp)
    j = rand(Categorical(exp.(logp - log_denom)))
    cluster = clusters[j]
    delete!(state.components, next)
    @assert next == next_new_cluster(state.crp)
    (cluster, log_denom)
end

function add_datum_conditional!(state::DPMMState, i::Int, datum::Float64)
    (cluster, log_denom) = conditional(state, i, datum)
    if cluster == next_new_cluster(state.crp)
        @assert !haskey(state.components, cluster)
        state.components[cluster] = NIGN()
    end
    incorporate!(state.components[cluster], datum)
    incorporate!(state.crp, i, cluster)
    log_denom
end

function remove_datum_conditional!(state::DPMMState, i::Int, datum::Float64)
    cluster = assignment(state.crp, i)
    unincorporate!(state.components[cluster], datum)
    if counts(state.crp, cluster) == 1
        delete!(state.components, cluster)
    end
    unincorporate!(state.crp, i)
    (cluster, log_denom) = conditional(state, i, datum)
    log_denom
end

function prior(state::DPMMState, i::Int, datum::Float64)
    @assert !has_assignment_for(state.crp, i)
    next = next_new_cluster(state.crp)
    cluster = draw!(state.crp, state.alpha, i)
    # the log_likelihood is p(y' | y_{others in cluster})
    if cluster == next
        component = NIGN()
        incorporate!(component, datum)
        log_likelihood = log_probability_density(component, state.nign_params)
    else
        old_ll = log_probability_density(state.components[cluster], state.nign_params)
        incorporate!(state.components[cluster], datum)
        new_ll = log_probability_density(state.components[cluster], state.nign_params)
        log_likelihood = new_ll - old_ll
        unincorporate!(state.components[cluster], datum)
    end
    unincorporate!(state.crp, i)
    @assert next == next_new_cluster(state.crp)
    (cluster, log_likelihood)
end

function add_datum_prior!(state::DPMMState, i::Int, datum::Float64)
    (cluster, log_likelihood) = prior(state, i, datum)
    if cluster == next_new_cluster(state.crp)
        @assert !haskey(state.components, cluster)
        state.components[cluster] = NIGN()
    end
    incorporate!(state.components[cluster], datum)
    incorporate!(state.crp, i, cluster)
    log_likelihood 
end

function remove_datum_prior!(state::DPMMState, i::Int, datum::Float64)
    cluster = assignment(state.crp, i)
    after_ll = log_probability_density(state.components[cluster], state.nign_params)
    unincorporate!(state.components[cluster], datum)
    before_ll = log_probability_density(state.components[cluster], state.nign_params)
    if counts(state.crp, cluster) == 1
        delete!(state.components, cluster)
    end
    unincorporate!(state.crp, i)
    log_likelihood = after_ll - before_ll
    log_likelihood
end

function gibbs_step!(state::DPMMState, i::Int, datum::Float64)
    remove_datum_conditional!(state, i, datum)
    add_datum_conditional!(state,i, datum)
end

function alpha_mh_update!(state::DPMMState)
    alpha_proposed = rand(state.hypers.alpha_prior)
    accept_ratio = log_probability(state.crp, alpha_proposed)
    accept_ratio -= log_probability(state.crp, state.alpha)
    if log(rand()) < accept_ratio
        # accept
        state.alpha = alpha_proposed
    end
end

function nign_params_logpdf(state::DPMMState)
    sum(cluster_likelihoods(state))
end

function nign_params_update!(state::DPMMState)
    accept_ratio = -nign_params_logpdf(state)
    m_old = state.nign_params.m
    r_old = state.nign_params.r
    nu_old = state.nign_params.nu
    s_old = state.nign_params.s
    state.nign_params.m = rand(state.hypers.nign_params_prior.m_prior)
    state.nign_params.r = rand(state.hypers.nign_params_prior.r_prior)
    state.nign_params.nu = rand(state.hypers.nign_params_prior.nu_prior)
    state.nign_params.s = rand(state.hypers.nign_params_prior.s_prior)
    accept_ratio += nign_params_logpdf(state)
    if log(rand()) >= accept_ratio
        # reject
        state.nign_params.m = m_old
        state.nign_params.r = r_old
        state.nign_params.nu = nu_old
        state.nign_params.s = s_old
    end
end

function nign_m_update!(state::DPMMState)
    accept_ratio = -nign_params_logpdf(state)
    m_old = state.nign_params.m
    state.nign_params.m = rand(state.hypers.nign_params_prior.m_prior)
    accept_ratio += nign_params_logpdf(state)
    if log(rand()) >= accept_ratio
        # reject
        state.nign_params.m = m_old
    end
end

function nign_r_update!(state::DPMMState)
    accept_ratio = -nign_params_logpdf(state)
    r_old = state.nign_params.r
    state.nign_params.r = rand(state.hypers.nign_params_prior.r_prior)
    accept_ratio += nign_params_logpdf(state)
    if log(rand()) >= accept_ratio
        # reject
        state.nign_params.r = r_old
    end
end

function nign_nu_update!(state::DPMMState)
    accept_ratio = -nign_params_logpdf(state)
    nu_old = state.nign_params.nu
    state.nign_params.nu = rand(state.hypers.nign_params_prior.nu_prior)
    accept_ratio += nign_params_logpdf(state)
    if log(rand()) >= accept_ratio
        # reject
        state.nign_params.nu = nu_old
    end
end

function nign_s_update!(state::DPMMState)
    accept_ratio = -nign_params_logpdf(state)
    s_old = state.nign_params.s
    state.nign_params.s = rand(state.hypers.nign_params_prior.s_prior)
    accept_ratio += nign_params_logpdf(state)
    if log(rand()) >= accept_ratio
        # reject
        state.nign_params.s = s_old
    end
end
