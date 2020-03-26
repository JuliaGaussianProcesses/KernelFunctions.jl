"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with length scale ell and period p. Given by
```math
    κ(x,y) =  h(x-z), h(t) = exp(-sum(t.^2./(ell.^2)))*cos(pi*sum(t./p))
```

"""
struct GaborKernel{T<:Real, K<:Kernel} <: BaseKernel
    ell::T
    p::T
    κ::K
    function GaborKernel(;ell::T=1.0, p::T=1.0) where {T<:Real}
        k = transform(SqExponentialKernel(), 1/ell)*transform(CosineKernel(), 1/p)
        new{T, typeof(k)}(ell, p, k)
    end
end

kappa(κ::GaborKernel, x, y) where {T<:Real} = kappa(κ.κ, x ,y)

function kernelmatrix(
    κ::GaborKernel,
    X::AbstractMatrix;
    obsdim::Int=defaultobs)
    kernelmatrix(κ.κ, X; obsdim=obsdim)
end

function kernelmatrix(
    κ::GaborKernel,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int=defaultobs)
    kernelmatrix(κ.κ, X, Y; obsdim=obsdim)
end

function kerneldiagmatrix(
    κ::GaborKernel,
    X::AbstractMatrix;
    obsdim::Int=defaultobs) #TODO Add test
    kerneldiagmatrix(κ.κ, X; obsdim=obsdim)
end
