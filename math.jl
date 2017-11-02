function logsumexp(logx::Vector{Float64})
    maxlog = maximum(logx)
    maxlog + log(sum(exp.(logx - maxlog)))
end

function logsumexp(logx::Matrix{Float64}, dim::Int)
    maxlog = maximum(logx, dim)
    maxlog + log(sum(exp.(logx .- maxlog), dim))
end
