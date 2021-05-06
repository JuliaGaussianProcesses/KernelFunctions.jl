"""
    gaborkernel(;
        sqexponential_transform=IdentityTransform(), cosine_tranform=IdentityTransform()
    )

Construct a Gabor kernel with transformations `sqexponential_transform` and
`cosine_transform` of the inputs of the underlying squared exponential and cosine kernel,
respectively.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the Gabor kernel with transformations ``f``
and ``g`` of the inputs to the squared exponential and cosine kernel, respectively,
is defined as
```math
k(x, x'; f, g) = \\exp\\bigg(- \\frac{\\| f(x) - f(x')\\|_2^2}{2}\\bigg)
                 \\cos\\big(\\pi \\|g(x) - g(x')\\|_2 \\big).
```
"""
function gaborkernel(;
    sqexponential_transform=IdentityTransform(), cosine_transform=IdentityTransform()
)
    return (SqExponentialKernel() ∘ sqexponential_transform) *
           (CosineKernel() ∘ cosine_transform)
end

# everything below will be removed
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

!!! note
    `GaborKernel` is deprecated and will be removed. Gabor kernels should be
    constructed with [`gaborkernel`](@ref) instead.
"""
struct GaborKernel{K<:Kernel} <: Kernel
    kernel::K

    function GaborKernel(; ell=nothing, p=nothing)
        Base.depwarn(
            "`GaborKernel` is deprecated and will be removed. Gabor kernels should be " *
            "constructed with `gaborkernel` instead.",
            :GaborKernel,
        )
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

_lengthscale(x) = 1
_lengthscale(k::TransformedKernel) = _lengthscale(k.transform)
_lengthscale(t::ScaleTransform) = inv(first(t.s))
_lengthscale(t::ARDTransform) = map(inv, t.v)

function Base.getproperty(k::GaborKernel, v::Symbol)
    if v == :kernel
        return getfield(k, v)
    elseif v == :ell
        return _lengthscale(k.kernel.kernels[1])
    elseif v == :p
        return _lengthscale(k.kernel.kernels[2])
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
