
"""Get method for the kernel metric"""
@inline metric(κ::Kernel) = κ.metric
"""Apply functions of a kernel on a distance"""
# @inline (κ::K)(d::Real) where {K<:Kernel} = kappa(κ,d)

@inline transform(κ::Kernel) = κ.transform
@inline transform(κ::Kernel,x::AbstractVecOrMat) = transform(κ.transform,x)
@inline transform(κ::Kernel,x::AbstractVecOrMat,obsdim::Int) = transform(κ.transform,x,obsdim)

# @inline (κ::Kernel)(x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = kappa(κ,evaluate(κ.(metric),x,y))
