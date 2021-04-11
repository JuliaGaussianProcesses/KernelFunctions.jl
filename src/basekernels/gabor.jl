"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with lengthscale `ell` and period `p`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Gabor kernel with lengthscale ``l_i > 0``
and period ``p_i > 0`` is defined as
```math
k(x, x'; l, p) = \\exp\\bigg(- \\cos\\bigg(\\pi\\sum_{i=1}^d \\frac{x_i - x'_i}{p_i}\\bigg)
                             \\sum_{i=1}^d \\frac{(x_i - x'_i)^2}{l_i^2}\\bigg).
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
    ell_transform = if ell === nothing
        IdentityTransform()
    elseif ell isa Real
        ScaleTransform(inv(ell))
    else
        ARDTransform(inv.(ell))
    end
    p_transform = if p === nothing
        IdentityTransform()
    elseif p isa Real
        ScaleTransform(inv(p))
    else
        ARDTransform(inv.(p))
    end

    return (SqExponentialKernel() ∘ ell_transform) * (CosineKernel() ∘ p_transform)
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

kernelmatrix_diag(κ::GaborKernel, x::AbstractVector) = kernelmatrix_diag(κ.kernel, x)

function kernelmatrix_diag(κ::GaborKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix_diag(κ.kernel, x, y)
end
