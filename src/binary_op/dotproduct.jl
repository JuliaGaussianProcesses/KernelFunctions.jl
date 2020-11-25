struct DotProduct end
# struct DotProduct <: Distances.UnionSemiMetric end


(d::DotProduct)(a, b) = evaluate(d, a, b)
Distances.evaluate(::DotProduct, a, b) = dot(a, b)

function Distances.pairwise(d::DotProduct, a::AbstractMatrix, b::AbstractMatrix=a; dims=1)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    m = size(a, dims)
    n = size(b, dims)
    P = Matrix{Distances.result_type(d, a, b)}(undef, m, n)
    if dims == 1
        return _pairwise!(P, d, transpose(a), transpose(b))
    else
        return _pairwise!(P, d, a, b)
    end
end

function Distances.pairwise!(P::AbstractMatrix, ::DotProduct, a::AbstractMatrix, b::AbstractMatrix=a; dims=1)
    dims = deprecated_dims(dims)
    dims in (1, 2) || throw(ArgumentError("dims should be 1 or 2 (got $dims)"))
    if dims == 1
        na, ma = size(a)
        nb, mb = size(b)
        ma == mb || throw(DimensionMismatch("The numbers of columns in a and b " *
                                            "must match (got $ma and $mb)."))
    else
        ma, na = size(a)
        mb, nb = size(b)
        ma == mb || throw(DimensionMismatch("The numbers of rows in a and b " *
                                            "must match (got $ma and $mb)."))
    end
    size(P) == (na, nb) ||
        throw(DimensionMismatch("Incorrect size of P (got $(size(P)), expected $((na, nb)))."))
    if dims == 1
        _pairwise!(P, metric, transpose(a), transpose(b))
    else
        _pairwise!(P, metric, a, b)
    end
end

function Distances._pairwise!(P::AbstractMatrix, ::DotProduct, a::AbstractMatrix, b::AbstractMatrix=a)
    for ij in CartesianIndices(P)
        P[ij] = @views dot(a[:, ij[1]], b[:, ij[2]])
    end
    return P
end

# @inline function Distances._evaluate(::DotProduct, a::AbstractVector, b::AbstractVector)
#     @boundscheck if length(a) != length(b)
#         throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
#     end
#     return dot(a,b)
# end

# Distances.result_type(::DotProduct, Ta::Type, Tb::Type) = promote_type(Ta, Tb)

# @inline Distances.eval_op(::DotProduct, a::Real, b::Real) = a * b
# @inline (dist::DotProduct)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist, a, b)
# @inline (dist::DotProduct)(a::Number,b::Number) = a * b
