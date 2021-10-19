# Add our own pairwise function to be able to apply it on vectors

function pairwise(d::BinaryOp, X::AbstractVector, Y::AbstractVector=X)
    return @tullio out[i, j] := d(X[i], Y[j])
end

function pairwise!(out::AbstractMatrix, d::BinaryOp, X::AbstractVector, Y::AbstractVector=X)
    return @tullio out[i, j] = d(X[i], Y[j])
end

# Also defines the colwise method for abstractvectors
# We have different methods for PreMetric and AbstractBinaryOp
# Since colwise on AbstractBinaryOp is not guaranteed to be equal to 0
function colwise(d::Distances.PreMetric, x::AbstractVector)
    return zeros(Distances.result_type(d, x, x), length(x)) # Valid since d(x,x) == 0 by definition
end

function colwise(d::Distances.PreMetric, x::VecOfVecs)
    return zeros(Distances.result_type(d, x.X, x.X), length(x)) # Valid since d(x,x) == 0 by definition
end

function colwise(d::AbstractBinaryOp, x::AbstractVector)
    return @tullio out[i] := d(x[i], x[i])
end

function colwise(d::BinaryOp, x::AbstractVector, y::AbstractVector)
    return @tullio out[i] := d(x[i], y[i])
end
