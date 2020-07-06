# Add our own pairwise function to be able to apply it on vectors

pairwise(d::PreMetric, X::AbstractVector, Y::AbstractVector) = broadcast(d, X, Y')

pairwise(d::PreMetric, X::AbstractVector) = pairwise(d, X, X)

function pairwise!(
    out::AbstractMatrix,
    d::PreMetric,
    X::AbstractVector,
    Y::AbstractVector,
)
    broadcast!(d, out, X, Y')
end

pairwise!(out::AbstractMatrix, d::PreMetric, X::AbstractVector) = pairwise!(out, d, X, X)

function pairwise(d::PreMetric, x::AbstractVector{<:Real})
    return Distances.pairwise(d, reshape(x, :, 1); dims = 1)
end

function pairwise(
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return Distances.pairwise(d, reshape(x, :, 1), reshape(y, :, 1); dims = 1)
end

function pairwise!(out::AbstractMatrix, d::PreMetric, x::AbstractVector{<:Real})
    return Distances.pairwise!(out, d, reshape(x, :, 1); dims = 1)
end

function pairwise!(
    out::AbstractMatrix,
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return Distances.pairwise!(out, d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end
