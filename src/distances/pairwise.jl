# Add our own pairwise function to be able to apply it on vectors

function pairwise(d::PreMetric, X::AbstractVector, Y::AbstractVector)
    return broadcast(d, X, permutedims(Y))
end

pairwise(d::PreMetric, X::AbstractVector) = pairwise(d, X, X)

function pairwise!(out::AbstractMatrix, d::PreMetric, X::AbstractVector, Y::AbstractVector)
    return broadcast!(d, out, X, Y')
end

pairwise!(out::AbstractMatrix, d::PreMetric, X::AbstractVector) = pairwise!(out, d, X, X)

function pairwise(d::PreMetric, x::AbstractVector{<:Real})
    return Distances.pairwise(d, reshape(x, :, 1); dims=1)
end

function pairwise(d::PreMetric, x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    return Distances.pairwise(d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end

function pairwise!(out::AbstractMatrix, d::PreMetric, x::AbstractVector{<:Real})
    return Distances.pairwise!(out, d, reshape(x, :, 1); dims=1)
end

function pairwise!(
    out::AbstractMatrix, d::PreMetric, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return Distances.pairwise!(out, d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end


# Also defines the colwise method for abstractvectors


function Distances.colwise(d::PreMetric, x::ColVecs)
    Distances.colwise(d, x.X, x.X)
end

function Distances.colwise(d::PreMetric, x::RowVecs)
    Distances.colwise(d, x.X', x.X')
end

function Distances.colwise(d::PreMetric, x::AbstractVector)
    d.(x, x)
end

function Distances.colwise(d::PreMetric, x::ColVecs, y::ColVecs)
    Distances.colwise(d, x.X, y.X)
end

function Distances.colwise(d::PreMetric, x::RowVecs, y::RowVecs)
    Distances.colwise(d, x.X', y.X')
end

function Distances.colwise(d::PreMetric, x::AbstractVector, y::AbstractVector)
    d.(x, y)
end