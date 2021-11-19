## DotProduct is not following the PreMetric rules since d(x, x) != 0 and d(x, y) >= 0 for all x, y
struct DotProduct <: Distances.UnionPreMetric end

@inline function Distances._evaluate(::DotProduct, a::AbstractVector, b::AbstractVector)
    @boundscheck if length(a) != length(b)
        throw(
            DimensionMismatch(
                "first array has length $(length(a)) which does not match the length of the second, $(length(b)).",
            ),
        )
    end
    return dot(a, b)
end

Distances.result_type(::DotProduct, Ta::Type, Tb::Type) = promote_type(Ta, Tb)

@inline Distances.eval_op(::DotProduct, a::Real, b::Real) = a * b
@inline function (dist::DotProduct)(a::AbstractArray, b::AbstractArray)
    return Distances._evaluate(dist, a, b)
end
@inline (dist::DotProduct)(a::Number, b::Number) = a * b
