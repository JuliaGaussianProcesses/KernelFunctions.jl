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
        ell_transform = _lengthscale_transform(ell)
        p_transform = _lengthscale_transform(p)
        k = (SqExponentialKernel() ∘ ell_transform) * (CosineKernel() ∘ p_transform)
        return new{typeof(k)}(k)
    end
end

@functor GaborKernel

(κ::GaborKernel)(x, y) = κ.kernel(x, y)

_lengthscale_transform(::Nothing) = IdentityTransform()
_lengthscale_transform(x::Real) = ScaleTransform(inv(x))
_lengthscale_transform(x::AbstractVector) = ARDTransform(map(inv, x))

_lengthscale(::IdentityTransform) = 1
_lengthscale(t::ScaleTransform) = inv(first(t.s))
_lengthscale(t::ARDTransform) = map(inv, t.v)

function Base.getproperty(k::GaborKernel, v::Symbol)
    if v == :kernel
        return getfield(k, v)
    elseif v == :ell
        ell_transform = k.kernel.kernels[1].transform
        return _lengthscale(ell_transform)
    elseif v == :p
        p_transform = k.kernel.kernels[2].transform
        return _lengthscale(p_transform)
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
