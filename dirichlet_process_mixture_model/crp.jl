import DataStructures.Stack

type CRPState
    # map from data index to cluster index
    assignments::Dict{Int, Int}

    # map from cluster index size of cluser
    counts::Dict{Int, Int}

    # reuse ids for the new clusters by pushing them onto a stack
    # this is necessary because we may do billions of Gibbs sweeps
    free::Stack{Int}

    # the next cluster id to allocate, after free stack is empty
    next_cluster::Int

    # create an empty CRP state
    function CRPState()
        free = Stack(Int)
        push!(free, 1)
        new(Dict{Int, Int}(), Dict{Int, Int}(), free, 2)
    end
end

has_assignment_for(crp::CRPState, i::Int) = haskey(crp.assignments, i)
next_new_cluster(crp::CRPState) = DataStructures.top(crp.free)
assignment(crp::CRPState, i::Int) = crp.assignments[i]
num_assignments(crp::CRPState) = length(crp.assignments)
has_cluster(crp::CRPState, cluster::Int) = haskey(crp.counts, cluster)
counts(crp::CRPState, cluster::Int) = crp.counts[cluster]
clusters(crp::CRPState) = keys(crp.counts)

function log_probability(crp::CRPState, alpha::Float64)
    N = length(crp.assignments)
    ll = length(crp.counts) * log(alpha)
    ll += sum(lgamma.(collect(values(crp.counts))))
    ll += lgamma(alpha) - lgamma(N + alpha)
    ll
end

function incorporate!(crp::CRPState, i::Int, cluster::Int)
    @assert !haskey(crp.counts, next_new_cluster(crp))
    @assert !haskey(crp.assignments, i)
    @assert (cluster == next_new_cluster(crp)) || haskey(crp.counts, cluster)
    crp.assignments[i] = cluster
    if cluster == next_new_cluster(crp)
        # allocate a new cluster
        crp.counts[cluster] = 0
        pop!(crp.free)
        if isempty(crp.free)
            @assert !haskey(crp.counts, crp.next_cluster)
            push!(crp.free, crp.next_cluster)
            crp.next_cluster += 1
        end
    else
        @assert crp.counts[cluster] > 0
    end
    crp.counts[cluster] += 1
end

function unincorporate!(crp::CRPState, i::Int)
    @assert !haskey(crp.counts, next_new_cluster(crp))
    @assert haskey(crp.assignments, i)
    cluster = crp.assignments[i]
    delete!(crp.assignments, i)
    crp.counts[cluster] -= 1
    if crp.counts[cluster] == 0
        # free the empyt cluster
        delete!(crp.counts, cluster)
        push!(crp.free, cluster)
    end
end

function draw!(crp::CRPState, alpha::Float64, data_index::Int)
    @assert !haskey(crp.assignments, data_index)
    clusters = collect(keys(crp.counts))
    probs = Array{Float64,1}(length(clusters) + 1)
    for (j, cluster) in enumerate(clusters)
        probs[j] = crp.counts[cluster]
    end
    probs[end] = alpha
    probs = probs / sum(probs)
    j = rand(Categorical(probs))
    if (j == length(clusters) + 1)
        # new cluster
        cluster = next_new_cluster(crp)
    else
        cluster = clusters[j]
    end
    incorporate!(crp, data_index, cluster)
    cluster
end
