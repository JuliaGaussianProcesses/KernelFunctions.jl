struct Delta <: Distances.PreMetric
end

@inline function Distances._evaluate(::Delta,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    return a == b
end

@inline (dist::Delta)(a::AbstractArray, b::AbstractArray) = Distances._evaluate(dist, a, b)
@inline (dist::Delta)(a::Number,b::Number) = a == b
