"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with lengthscale `ell` and period `p`. Given by
```math
    κ(x,y) =  h(x-z), h(t) = exp(-sum(t.^2./(ell.^2)))*cos(pi*sum(t./p))
```

"""
struct GaborKernel{K<:Kernel} <: Kernel
    kernel::K
    function GaborKernel(;ell=nothing, p=nothing)
        k = _gabor(ell=ell, p=p)
        return new{typeof(k)}(k)
    end
end

@functor GaborKernel

(κ::GaborKernel)(x, y) = κ.kernel(x ,y)

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
    if v == :kernel
        return getfield(k, v)
    elseif v == :ell
        kernel1 = k.kernel.kernels[1]
    	if kernel1 isa TransformedKernel
    	    return 1 ./ kernel1.transform.s[1]
    	else
    	    return 1.0
    	end
    elseif v == :p
        kernel2 = k.kernel.kernels[2]
        if kernel2 isa TransformedKernel
            return 1 ./ kernel2.transform.s[1]
        else
            return 1.0
        end
    else
        error("Invalid Property")
    end
end

Base.show(io::IO, κ::GaborKernel) = print(io, "Gabor Kernel (ell = ", κ.ell, ", p = ", κ.p, ")")

function kernelmatrix(κ::GaborKernel, X::AbstractMatrix; obsdim::Int=defaultobs)
    return kernelmatrix(κ.kernel, X; obsdim=obsdim)
end

function kernelmatrix(
    κ::GaborKernel, X::AbstractMatrix, Y::AbstractMatrix;
    obsdim::Int=defaultobs,
)
    return kernelmatrix(κ.kernel, X, Y; obsdim=obsdim)
end

function kerneldiagmatrix(κ::GaborKernel, X::AbstractMatrix; obsdim::Int=defaultobs) #TODO Add test
    return kerneldiagmatrix(κ.kernel, X; obsdim=obsdim)
end
