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

metric(::SqExponentialKernel) = SqEuclidean()

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

metric(::ExponentialKernel) = Euclidean()

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

!!! warning
    The default value of parameter `γ` will be changed to `1.0` in the next breaking release
    of KernelFunctions.

See also: [`ExponentialKernel`](@ref), [`SqExponentialKernel`](@ref)

[^RW]: C. E. Rasmussen & C. K. I. Williams (2006). Gaussian Processes for Machine Learning.
"""
struct GammaExponentialKernel{Tγ<:Real} <: SimpleKernel
    γ::Vector{Tγ}
    # function GammaExponentialKernel(; gamma::Real=1.0, γ::Real=gamma)
    function GammaExponentialKernel(; gamma=nothing, γ=gamma)
        γ2 = if γ === nothing
            Base.depwarn(
                "the default value of parameter `γ` of the `GammaExponentialKernel` will " *
                "be changed to `1.0` in the next breaking release of KernelFunctions",
                :GammaExponentialKernel,
            )
            2.0
        else
            γ
        end
        @check_args(GammaExponentialKernel, γ2, zero(γ2) < γ2 ≤ 2, "γ ∈ (0, 2]")
        return new{typeof(γ2)}([γ2])
    end
end

@functor GammaExponentialKernel

kappa(κ::GammaExponentialKernel, d::Real) = exp(-d^first(κ.γ))

metric(::GammaExponentialKernel) = Euclidean()

iskroncompatible(::GammaExponentialKernel) = true

function Base.show(io::IO, κ::GammaExponentialKernel)
    return print(io, "Gamma Exponential Kernel (γ = ", first(κ.γ), ")")
end
