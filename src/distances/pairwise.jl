# Add the possibility to use Distances.pairwise on vectors as well

function Distances.pairwise(
    d::PreMetric, 
    X::AbstractVector, 
    Y::AbstractVector,
)
    broadcast(d, X, Y')
end

Distances.pairwise(d::PreMetric, X::AbstractVector) = pairwise(d, X, X)

function Distances.pairwise!(
    out::AbstractMatrix, 
    d::PreMetric, 
    X::AbstractVector, 
    Y::AbstractVector,
)
    broadcast!(d, out, X, Y')
end

Distances.pairwise!(out::AbstractMatrix, d::PreMetric, X::AbstractVector) = pairwise!(out, d, X, X)

# This is type piracy. We should not doing this.
function Distances.pairwise(d::PreMetric, x::AbstractVector{<:Real})
    return pairwise(d, reshape(x, :, 1); dims=1)
end

function Distances.pairwise(
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return pairwise(d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end

function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::AbstractVector{<:Real})
    return pairwise!(out, d, reshape(x, :, 1); dims=1)
end

function Distances.pairwise!(
    out::AbstractMatrix,
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return pairwise!(out, d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end