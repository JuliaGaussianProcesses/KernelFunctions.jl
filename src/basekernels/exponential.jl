"""
    SqExponentialKernel()

Squared exponential kernel.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the squared exponential kernel is defined as
```math
k(x, x') = \\exp\\bigg(- \\frac{\\|x - x'\\|_2^2}{2}\\bigg).
```

See also: [`GammaExponentialKernel`](@ref)
"""
struct SqExponentialKernel <: SimpleKernel end

kappa(::SqExponentialKernel, d²::Real) = exp(-d² / 2)

binary_op(::SqExponentialKernel) = SqEuclidean()

iskroncompatible(::SqExponentialKernel) = true

Base.show(io::IO, ::SqExponentialKernel) = print(io, "Squared Exponential Kernel")

## Aliases ##

"""
    RBFKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const RBFKernel = SqExponentialKernel

"""
    GaussianKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const GaussianKernel = SqExponentialKernel

"""
    SEKernel()

Alias of [`SqExponentialKernel`](@ref).
"""
const SEKernel = SqExponentialKernel

"""
    ExponentialKernel()

Exponential kernel.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the exponential kernel is defined as
```math
k(x, x') = \\exp\\big(- \\|x - x'\\|_2\\big).
```

See also: [`GammaExponentialKernel`](@ref)
"""
struct ExponentialKernel <: SimpleKernel end

kappa(::ExponentialKernel, d::Real) = exp(-d)

binary_op(::ExponentialKernel) = Euclidean()

iskroncompatible(::ExponentialKernel) = true

Base.show(io::IO, ::ExponentialKernel) = print(io, "Exponential Kernel")

## Aliases ##

"""
    LaplacianKernel()

Alias of [`ExponentialKernel`](@ref).
"""
const LaplacianKernel = ExponentialKernel

"""
    Matern12Kernel()

Alias of [`ExponentialKernel`](@ref).
"""
const Matern12Kernel = ExponentialKernel

"""
    GammaExponentialKernel(; γ::Real=2.0)

γ-exponential kernel with parameter `γ`.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the γ-exponential kernel[^RW] with parameter
``\\gamma \\in (0, 2]`` is defined as
```math
k(x, x'; \\gamma) = \\exp\\big(- \\|x - x'\\|_2^{\\gamma}\\big).
```

See also: [`ExponentialKernel`](@ref), [`SqExponentialKernel`](@ref)

[^RW]: C. E. Rasmussen & C. K. I. Williams (2006). Gaussian Processes for Machine Learning.
"""
struct GammaExponentialKernel{Tγ<:Real} <: SimpleKernel
    γ::Vector{Tγ}
    function GammaExponentialKernel(; gamma::Real=2.0, γ::Real=gamma)
        @check_args(GammaExponentialKernel, γ, zero(γ) < γ ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(γ)}([γ])
    end
end

@functor GammaExponentialKernel

kappa(κ::GammaExponentialKernel, d::Real) = exp(-d^first(κ.γ))

binary_op(::GammaExponentialKernel) = Euclidean()

iskroncompatible(::GammaExponentialKernel) = true

function Base.show(io::IO, κ::GammaExponentialKernel)
    return print(io, "Gamma Exponential Kernel (γ = ", first(κ.γ), ")")
end
