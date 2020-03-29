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
    function GaborKernel(;ell=nothing, p=nothing)
        k = _gabor(ell=ell, p=p)
        if ell==nothing ell=1.0 end
        if p==nothing p=1.0 end
        new{Union{typeof(ell),typeof(p)}, typeof(k)}(ell, p, k)
    end
end

function _gabor(; ell = nothing, p = nothing)
    if ell === nothing
        if p === nothing
            return SqExponentialKernel() * CosineKernel()
        else
            return SqExponentialKernel() * transform(CosineKernel(), 1 ./ p)
        end
    elseif p === nothing
        return transform(SqExponentialKernel(), 1 ./ ell) * CosineKernel()
    else
        return transform(SqExponentialKernel(), 1 ./ ell) * transform(CosineKernel(), 1 ./ p)
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
