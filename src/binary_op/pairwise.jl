# Add our own pairwise function to be able to apply it on vectors

pairwise(d::BinaryOp, X::AbstractVector, Y::AbstractVector) = broadcast(d, X, permutedims(Y))

pairwise(d::BinaryOp, X::AbstractVector) = pairwise(d, X, X)

function pairwise!(
    out::AbstractMatrix,
    d::BinaryOp,
    X::AbstractVector,
    Y::AbstractVector,
)
    broadcast!(d, out, X, Y')
end

pairwise!(out::AbstractMatrix, d::BinaryOp, X::AbstractVector) = pairwise!(out, d, X, X)

function pairwise(d::BinaryOp, x::AbstractVector{<:Real})
    return Distances.pairwise(d, reshape(x, :, 1); dims = 1)
end

function pairwise(
    d::BinaryOp,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return Distances.pairwise(d, reshape(x, :, 1), reshape(y, :, 1); dims = 1)
end

function pairwise!(out::AbstractMatrix, d::BinaryOp, x::AbstractVector{<:Real})
    return Distances.pairwise!(out, d, reshape(x, :, 1); dims = 1)
end

function pairwise!(
    out::AbstractMatrix,
    d::BinaryOp,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return Distances.pairwise!(out, d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end

