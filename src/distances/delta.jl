struct Delta <: AbstractBinaryOp end

# Basic definitions
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
