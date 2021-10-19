# Add our own pairwise function to be able to apply it on vectors

function pairwise(d::BinaryOp, X::AbstractVector, Y::AbstractVector)
    return @tullio out[i, j] := d(X[i], Y[j])
end

pairwise(d::BinaryOp, X::AbstractVector) = pairwise(d, X, X)

function pairwise!(out::AbstractMatrix, d::BinaryOp, X::AbstractVector, Y::AbstractVector)
    return @tullio out[i, j] = d(X[i], Y[j])
end

pairwise!(out::AbstractMatrix, d::BinaryOp, X::AbstractVector) = pairwise!(out, d, X, X)

# Also defines the colwise method for abstractvectors

function colwise(d::PreMetric, x::AbstractVector)
    return zeros(Distances.result_type(d, x, x), length(x)) # Valid since d(x,x) == 0 by definition
end

function colwise(d::PreMetric, x::VecOfVecs)
    return zeros(Distances.result_type(d, x.X, x.X), length(x)) # Valid since d(x,x) == 0 by definition
end

function colwise(d::AbstractBinaryOp, x::AbstractVector)
    return @tullio out[i] := d(x[i], x[i])
end

function colwise(d::PreMetric, x::AbstractVector, y::AbstractVector)
    return @tullio out[i] := d(x[i], y[i])
end