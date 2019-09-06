
@inline metric(κ::Kernel) = κ.metric
@inline (κ::K)(d::Real) where {K<:Kernel} = kappa(κ,d)

### Transform generics

@inline transform(κ::Kernel) = κ.transform
@inline transform(κ::Kernel,x::AbstractVecOrMat) = transform(κ.transform,x)
@inline transform(κ::Kernel,x::AbstractVecOrMat,obsdim::Int) = transform(κ.transform,x,obsdim)

@inline (κ::Kernel)(x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = kernel(κ,evaluate(κ.(metric),x,y))
