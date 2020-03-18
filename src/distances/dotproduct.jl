struct DotProduct <: Distances.SemiMetric end

@inline function Distances._evaluate(::DotProduct, a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    @boundscheck if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    return dot(a,b)
end

# For later convenience once Distances.jl open their API
@inline Distances.eval_op(::DotProduct, a::Real, b::Real) = a * b

@inline (dist::DotProduct)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist, a, b)
@inline (dist::DotProduct)(a::Number,b::Number) = Distances.eval_op(dist, a, b)
