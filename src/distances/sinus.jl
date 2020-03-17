struct Sinus{T} <: Distances.PreMetric
    r::Vector{T}
end

@inline function Distances._evaluate(d::Sinus,a::AbstractVector{T},b::AbstractVector{T}) where {T}
    @boundscheck if (length(a) != length(b)) || length(a) != length(d.r)
        throw(DimensionMismatch("Dimensions of the inputs are not matching : a = $(length(a)), b = $(length(b)), r = $(length(d.r))"))
    end
    return sum(abs2,sin.(π.*(a-b))./d.r)
end

@inline (dist::Sinus)(a::AbstractArray,b::AbstractArray) = Distances._evaluate(dist,a,b)
@inline (dist::Sinus)(a::Number,b::Number) = (sin(π*(a-b))/first(dist.r))^2
