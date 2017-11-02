using SMC
import JSON

function load_states(fname::String)
    states = JSON.parsefile(fname)
    return convert(Vector{Int}, states)
end

function load_observations(fname::String)
    observations = JSON.parsefile(fname)
    return convert(Vector{Int}, observations)
end


"""
Load matrix from column-major format
"""
function to_float_matrix(vec::Vector, num_rows::Int, num_cols::Int)
    if length(vec) != num_cols
        error("Dimension mismatch: vector has length $(length(vec)) but expected $num_cols")
    end
    mat = fill(NaN, (num_rows, num_cols))
    for (i, column) in enumerate(vec)
        if length(column) != num_rows
            error("Dimension mismatch")
        end
        mat[:,i] = column
    end
    if any(isnan.(mat)[:])
        error("Dimension mismatch")
    end
    return mat
end

function load_hmm(fname::String)
    dict = JSON.parsefile(fname)
    num_states = dict["num_states"]
    num_obs = dict["num_obs"]
    initial_state_prior = dict["initial_state_prior"]
    initial_state_prior = convert(Vector{Float64}, initial_state_prior)
    if length(initial_state_prior) != num_states
        error("Dimension mismatch")
    end
    transition_model = dict["transition_model"]
    transition_model = to_float_matrix(transition_model, num_states, num_states)
    observation_model = dict["observation_model"]
    observation_model = to_float_matrix(observation_model, num_states, num_obs)
    log_initial_state_prior = dict["log_initial_state_prior"]
    log_transition_model = dict["log_transition_model"]
    log_observation_model = dict["log_observation_model"]
    return HiddenMarkovModel(initial_state_prior, transition_model, observation_model)
end
