abstract DPMMInitializer
#abstract type DPMMInitializer end

immutable PriorProposalDPMMInitializer <: DPMMInitializer
    datum::Float64 # the first data point in the data set 
    hypers::Hyperparameters
end

function add_first_datum!(init::PriorProposalDPMMInitializer, state::DPMMState)
    add_datum_prior!(state, 1, init.datum)    
end

function remove_first_datum!(init::PriorProposalDPMMInitializer, state::DPMMState)
    remove_datum_prior!(state, 1, init.datum)
end

immutable OptimalProposalDPMMInitializer <: DPMMInitializer
    datum::Float64 # the first data point in the data set 
    hypers::Hyperparameters
end

function add_first_datum!(init::OptimalProposalDPMMInitializer, state::DPMMState)
    add_datum_conditional!(state, 1, init.datum)    
end

function remove_first_datum!(init::OptimalProposalDPMMInitializer, state::DPMMState)
    remove_datum_conditional!(state, 1, init.datum)
end

function SMC.forward(init::DPMMInitializer)
    alpha = rand(init.hypers.alpha_prior)
    nign_params = NIGNParams(init.hypers.nign_params_prior)
	state = DPMMState(alpha, CRPState(), OrderedDict{Int,NIGN}(), 
                      nign_params, init.hypers)
    log_weight = add_first_datum!(init, state)
	@assert num_assignments(state.crp) == 1
	@assert length(state.components) == 1
    (state, log_weight)
end

function SMC.backward(init::DPMMInitializer, state::DPMMState)
	@assert num_assignments(state.crp) == 1
	@assert length(state.components) == 1
    # we don't actually want to remove the datum from the state
    # TODO implement a second function that does't remove it
    tmp_state = deepcopy(state)
    log_weight = remove_first_datum!(init, tmp_state)
	@assert num_assignments(tmp_state.crp) == 0
	@assert length(tmp_state.components) == 0
    # TODO it seems strange that we don't need to check the prior probs on parameters?
    log_weight
end

abstract DPMMIncrementer
#abstract type DPMMIncrementer end

immutable PriorProposalDPMMIncrementer <: DPMMIncrementer
    data::Vector{Float64}
	datum_index::Int
    rejuvenation_sweeps::Int
end

function add_datum!(incr::PriorProposalDPMMIncrementer, state::DPMMState)
    add_datum_prior!(state, incr.datum_index, incr.data[incr.datum_index])
end

function remove_datum!(incr::PriorProposalDPMMIncrementer, state::DPMMState)
    remove_datum_prior!(state, incr.datum_index, incr.data[incr.datum_index])
end

immutable OptimalProposalDPMMIncrementer <: DPMMIncrementer
    data::Vector{Float64}
	datum_index::Int
    rejuvenation_sweeps::Int
end

function add_datum!(incr::OptimalProposalDPMMIncrementer, state::DPMMState)
    add_datum_conditional!(state, incr.datum_index, incr.data[incr.datum_index])
end

function remove_datum!(incr::OptimalProposalDPMMIncrementer, state::DPMMState)
    remove_datum_conditional!(state, incr.datum_index, incr.data[incr.datum_index])
end


function SMC.forward(incr::DPMMIncrementer, state::DPMMState)
	# sample the new clustering from the conditional distribution, 
    # and obtain the normalizing constant
    datum = incr.data[incr.datum_index]
	@assert num_assignments(state.crp) == incr.datum_index - 1
	new_state = deepcopy(state)

    # do a Rejuvenation sweep over the already incorporated data points
    for sweep=1:incr.rejuvenation_sweeps
        for i=1:incr.datum_index-1
            gibbs_step!(new_state, i, incr.data[i])
            alpha_mh_update!(new_state)
            nign_params_update!(new_state)
            nign_m_update!(new_state)
            nign_r_update!(new_state)
            nign_nu_update!(new_state)
            nign_s_update!(new_state)
        end
        nign_params_update!(new_state)
        nign_m_update!(new_state)
        nign_r_update!(new_state)
        nign_nu_update!(new_state)
        nign_s_update!(new_state)
    end

    log_weight = add_datum!(incr, new_state)
	@assert num_assignments(new_state.crp) == incr.datum_index
	(new_state, log_weight)
end

function SMC.backward(incr::DPMMIncrementer, new_state::DPMMState)
	# just get conditional sampling normalizing constant
    datum = incr.data[incr.datum_index]
	@assert num_assignments(new_state.crp) == incr.datum_index
	state = deepcopy(new_state)
    log_weight = remove_datum!(incr, state)
	@assert num_assignments(state.crp) == incr.datum_index - 1

    # Rejuvenation sweep (in reverse order)
    for sweep=1:incr.rejuvenation_sweeps
        nign_s_update!(state)
        nign_nu_update!(state)
        nign_r_update!(state)
        nign_m_update!(state)
        nign_params_update!(state)
        for i=incr.datum_index-1:1
            nign_s_update!(state)
            nign_nu_update!(state)
            nign_r_update!(state)
            nign_m_update!(state)
            nign_params_update!(state)
            alpha_mh_update!(state)
            gibbs_step!(state, i, incr.data[i])
        end
    end
	(state, log_weight)
end

function make_prior_proposal_smc_scheme(data::Vector{Float64}, hypers::Hyperparameters,
                                         rejuvenation_sweeps::Int, num_particles::Int)
    initializer = PriorProposalDPMMInitializer(data[1], hypers) 
    incrementers = Vector{PriorProposalDPMMIncrementer}(length(data)-1)
    for i=2:length(data)
        incrementers[i-1] = PriorProposalDPMMIncrementer(data, i, rejuvenation_sweeps)
    end
    SMCScheme(initializer, incrementers, num_particles)
end

function make_prior_proposal_smc_sampler(data::Vector{Float64}, hypers::Hyperparameters,
                                         rejuvenation_sweeps::Int, num_particles::Int)
    scheme = make_prior_proposal_smc_scheme(data, hypers, rejuvenation_sweeps, num_particles)
    SMCSampler(scheme, log_joint_density)
end

function make_optimal_proposal_smc_scheme(data::Vector{Float64}, hypers::Hyperparameters,
                                           rejuvenation_sweeps::Int, num_particles::Int)
    initializer = OptimalProposalDPMMInitializer(data[1], hypers) 
    incrementers = Vector{OptimalProposalDPMMIncrementer}(length(data)-1)
    for i=2:length(data)
        incrementers[i-1] = OptimalProposalDPMMIncrementer(data, i, rejuvenation_sweeps)
    end
    SMCScheme(initializer, incrementers, num_particles)
end

function make_optimal_proposal_smc_sampler(data::Vector{Float64}, hypers::Hyperparameters,
                                           rejuvenation_sweeps::Int, num_particles::Int)
    scheme = make_optimal_proposal_smc_scheme(data, hypers, rejuvenation_sweeps, num_particles)
    SMCSampler(scheme, log_joint_density)
end
