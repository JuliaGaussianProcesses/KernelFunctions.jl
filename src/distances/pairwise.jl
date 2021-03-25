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

function colwise(d::PreMetric, x::AbstractVector)
    return zeros(Distances.result_type(d, x, x), length(x)) # Valid since d(x,x) == 0 by definition
end

## The following is a hack for DotProduct and Delta to still work
function colwise(d::Distances.UnionPreMetric, x::ColVecs)
    return Distances.colwise(d, x.X, x.X)
end

function colwise(d::Distances.UnionPreMetric, x::RowVecs)
    return Distances.colwise(d, x.X', x.X')
end

function colwise(d::Distances.UnionPreMetric, x::AbstractVector)
    return map(d, x, x)
end

function colwise(d::PreMetric, x::ColVecs, y::ColVecs)
    return Distances.colwise(d, x.X, y.X)
end

function colwise(d::PreMetric, x::RowVecs, y::RowVecs)
    return Distances.colwise(d, x.X', y.X')
end

function colwise(d::PreMetric, x::AbstractVector, y::AbstractVector)
    return map(d, x, y)
end
