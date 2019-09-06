
@inline metric(κ::Kernel) = κ.metric
kernels =
for k in [:SquaredExponentialKernel,:MaternKernel,:Matern32Kernel,:Matern52Kernel]
    eval(quote
        @inline (κ::$k)(d::Real) = kappa(κ,d)
        @inline (κ::$k)(x::AbstractVector{T},y::AbstractVector{T}) where {T} = kernel(κ,evaluate(κ.(metric),x,y))
        @inline (κ::$k)(x::AbstractMatrix{T},y::AbstractMatrix{T},obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,x,y,obsdim=obsdim)
    end)
end
### Transform generics

@inline transform(κ::Kernel) = κ.transform
@inline transform(κ::Kernel,x::AbstractVecOrMat) = transform(κ.transform,x)
@inline transform(κ::Kernel,x::AbstractVecOrMat,obsdim::Int) = transform(κ.transform,x,obsdim)
