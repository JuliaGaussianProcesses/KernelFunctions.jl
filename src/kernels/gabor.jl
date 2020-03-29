"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with length scale ell and period p. Given by
```math
    κ(x,y) =  h(x-z), h(t) = exp(-sum(t.^2./(ell.^2)))*cos(pi*sum(t./p))
```

"""
struct GaborKernel{T<:Real, K<:Kernel} <: BaseKernel
    kernel::K
    function GaborKernel(;ell=nothing, p=nothing)
        k = _gabor(ell=ell, p=p)
        new{typeof(k)}(k)
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

function Base.getproperty(k::GaborKernel, v::Symbol)
    if v == :κ 
        return getfield(k, v)
    elseif v == :ell
        kernel1 = k.κ.kernels[1]
    	if kernel1 isa TransformedKernel
    	    return 1 ./ kernel1.transform.s[1]
    	else
    	    return 1.0
    	end
    elseif v == :p
        kernel2 = k.κ.kernels[2]
        if kernel2 isa TransformedKernel
            return 1 ./ kernel2.transform.s[1]
        else
            return 1.0
        end
    else
        error("Invalid Property")
    end
end

kappa(κ::GaborKernel, x, y) = kappa(κ.κ, x ,y)

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
