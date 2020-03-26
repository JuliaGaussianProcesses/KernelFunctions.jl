struct Sinus{T} <: Distances.SemiMetric
    r::Vector{T}
end

Distances.parameters(d::Sinus) = d.r

@inline function Distances._evaluate(d::Sinus, a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    @boundscheck if (length(a) != length(b)) || length(a) != length(d.r)
        throw(DimensionMismatch("Dimensions of the inputs are not matching : a = $(length(a)), b = $(length(b)), r = $(length(d.r))"))
    end
    return sum(abs2, sinpi.(a - b) ./ d.r)
end

# For later convenience once Distances.jl open their API
@inline Distances.eval_op(::Sinus, a::Real, b::Real, p::Real) = abs2(sinpi(a - b) / p)

@inline (dist::Sinus)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist, a, b)
@inline (dist::Sinus)(a::Number,b::Number) = abs2(sinpi(a - b) / first(dist.r))
