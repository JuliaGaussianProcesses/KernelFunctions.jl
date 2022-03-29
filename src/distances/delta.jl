# Delta is not following the PreMetric rules since d(x, x) == 1
struct Delta <: Distances.UnionPreMetric end

(dist::Delta)(a::Number, b::Number) = a == b
Base.@propagate_inbounds function (dist::Delta)(
    a::AbstractArray{<:Number}, b::AbstractArray{<:Number}
)
    @boundscheck if length(a) != length(b)
        throw(
            DimensionMismatch(
                "first array has length $(length(a)) which does not match the length of the second, $(length(b)).",
            ),
        )
    end
    return a == b
end

Distances.result_type(::Delta, Ta::Type, Tb::Type) = Bool
