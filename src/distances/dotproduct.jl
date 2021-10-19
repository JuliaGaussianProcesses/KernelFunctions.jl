## DotProduct is not following the PreMetric rules since d(x, x) != 0 and d(x, y) >= 0 for all x, y
struct DotProduct <: AbstractBinaryOp end

(::DotProduct)(a::AbstractVector, b::AbstractVector) = dot(a, b)

(::DotProduct)(a::Number, b::Number) = a * b

function pairwise(::DotProduct, x::ColVecs, y::ColVecs)
    return @tullio out[i, j] := x.X[k, i] * y.X[k, j]
end

function pairwise(::DotProduct, x::RowVecs, y::RowVecs)
    return @tullio out[i, j] := x.X[i, k] * y.X[j, k]
end

function colwise(::DotProduct, x::RowVecs, y::RowVecs=x)
    return @tullio out[i] := x.X[i, k] * y.X[i, k]
end

function colwise(::DotProduct, x::ColVecs, y::ColVecs=x)
    return @tullio out[i] := x.X[k, i] * y.X[k, i]
end
