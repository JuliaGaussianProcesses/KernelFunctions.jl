"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with lengthscale `ell` and period `p`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Gabor kernel with lengthscale ``l_i > 0``
and period ``p_i > 0`` is defined as
```math
k(x, x'; l, p) = \\exp\\bigg(- \\sum_{i=1}^d \\frac{(x_i - x'_i)^2}{2l_i^2}\\bigg)
                 \\cos\\bigg(\\pi \\bigg(\\sum_{i=1}^d \\frac{(x_i - x'_i)^2}{p_i^2} \\bigg)^{1/2}\\bigg).
```
"""
struct GaborKernel{K<:Kernel} <: Kernel
    kernel::K
    function GaborKernel(; ell=nothing, p=nothing)
        k = _gabor(; ell=ell, p=p)
        return new{typeof(k)}(k)
    end
end

@functor GaborKernel

(κ::GaborKernel)(x, y) = κ.kernel(x, y)

function _gabor(; ell=nothing, p=nothing)
    if ell === nothing
        if p === nothing
            return SqExponentialKernel() * CosineKernel()
        else
            return SqExponentialKernel() * transform(CosineKernel(), 1 ./ p)
        end
    elseif p === nothing
        return transform(SqExponentialKernel(), 1 ./ ell) * CosineKernel()
    else
        return transform(SqExponentialKernel(), 1 ./ ell) *
               transform(CosineKernel(), 1 ./ p)
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

function Base.show(io::IO, κ::GaborKernel)
    return print(io, "Gabor Kernel (ell = ", κ.ell, ", p = ", κ.p, ")")
end

kernelmatrix(κ::GaborKernel, x::AbstractVector) = kernelmatrix(κ.kernel, x)

function kernelmatrix(κ::GaborKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(κ.kernel, x, y)
end

kerneldiagmatrix(κ::GaborKernel, x::AbstractVector) = kerneldiagmatrix(κ.kernel, x)
