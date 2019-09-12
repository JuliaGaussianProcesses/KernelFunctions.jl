struct DotProduct <: Distances.PreMetric
end

@inline function Distances._evaluate(::DotProduct,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    return dot(a,b)
end

@inline (dist::DotProduct)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist,a,b)
@inline (dist::DotProduct)(a::Number,b::Number) = a*b
