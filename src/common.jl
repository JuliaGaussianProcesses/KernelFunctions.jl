
"""Get method for the kernel metric"""
@inline metric(κ::Kernel) = κ.metric
"""Apply functions of a kernel on a distance"""
@inline (κ::Kernel)(d::Real) = kappa(κ,d)


@inline (κ::Kernel)(x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = kappa(κ,evaluate(κ.(metric),x,y))
