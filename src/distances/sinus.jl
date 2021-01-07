struct Sinus{T} <: Distances.SemiMetric
    # struct Sinus{T} <: Distances.UnionSemiMetric
    r::Vector{T}
end

Distances.parameters(d::Sinus) = d.r
@inline Distances.eval_op(::Sinus, a::Real, b::Real, p::Real) = abs2(sinpi(a - b) / p)
@inline (dist::Sinus)(a::AbstractArray, b::AbstractArray) = Distances._evaluate(dist, a, b)
@inline (dist::Sinus)(a::Number, b::Number) = abs2(sinpi(a - b) / first(dist.r))

Distances.result_type(::Sinus{T}, Ta::Type, Tb::Type) where {T} = promote_type(T, Ta, Tb)

@inline function Distances._evaluate(
    d::Sinus,
    a::AbstractVector,
    b::AbstractVector,
) where {T}
    @boundscheck if (length(a) != length(b)) || length(a) != length(d.r)
        throw(DimensionMismatch("Dimensions of the inputs are not matching : a = $(length(a)), b = $(length(b)), r = $(length(d.r))"))
    end
    return sum(abs2, sinpi.(a - b) ./ d.r)
end
