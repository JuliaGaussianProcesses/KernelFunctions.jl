@doc raw"""
    SpectralMixtureKernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractVector{<:Real},
        γ::AbstractMatrix{<:Real},
        ω::AbstractMatrix{<:Real},
    )
    SpectralMixtureKernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractVector{<:Real},
        γ::AbstractVector{<:AbstractVecOrMat{<:Real}},
        ω::AbstractVector{<:AbstractVecOrMat{<:Real}},
    )

Generalised Spectral Mixture kernel function as described in [1] in equation 6.
This family of functions is dense in the family of stationary real-valued kernels with respect to the pointwise convergence.[1]

## Definition

For inputs ``x, x′ \in \mathbb{R}^D``, the spectral mixture kernel ``\tilde{k}`` with ``K`` mixture components, mixture weights ``\alpha \in \mathbb{R}^K``, linear transformations ``\gamma_1, \ldots, \gamma_K \in \mathbb{R}^D``, and frequencies ``\omega_1, \ldots, \omega_K \in \mathbb{R}^D`` derived from a translation-invariant kernel ``k`` is defined as
```math
\tilde{k}(x, x'; \alpha, \gamma_1, \ldots, \gamma_K, \omega_1, \ldots, \omega_K, k) = \sum_{i=1}^K \alpha_i k(\gamma_i \odot x, \gamma_i \odot y) \cos(2\pi \omega_i^\top (x-y)).
```

## Arguments
- `h`: Stationary kernel (translation invariant), [`SqExponentialKernel`](@ref) by default
- `α`: Weight vector of each mixture component (should be positive)
- `γ`: Linear transformation of the input for `h`.
- `ω`: Frequencies for the cosine function. (should be positive)

`γ` and `ω` can be an
- `AbstractMatrix` of dimension `D x K` where `D` is the dimension of the inputs 
and `K` is the number of components
- `AbstractVector` of `K` `D`-dimensional `AbstractVector`


# References:
    [1] Generalized Spectral Kernels, by Yves-Laurent Kom Samo and Stephen J. Roberts
    [2] SM: Gaussian Process Kernels for Pattern Discovery and Extrapolation,
            ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams,
    [3] Covariance kernels for fast automatic pattern discovery and extrapolation
        with Gaussian processes, Andrew Gordon Wilson, PhD Thesis, January 2014.
        http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
    [4] http://www.cs.cmu.edu/~andrewgw/pattern/.

"""
struct SpectralMixtureKernel{
    K<:Kernel,Tα<:AbstractVector,Tγ<:AbstractVector,Tω<:AbstractVector
} <: Kernel
    kernel::K
    α::Tα
    γ::Tγ
    ω::Tω
    function SpectralMixtureKernel(
        h::Kernel,
        α::AbstractVector{<:Real},
        γ::AbstractVector{<:AbstractVector},
        ω::AbstractVector{<:AbstractVector},
    )
        (length(α) == length(γ) == length(ω)) ||
            throw(DimensionMismatch("The dimensions of α, γ, ans ω do not match"))
        any(<(0), α) && throw(ArgumentError("At least one element of α is negative"))
        any(any.(<(0), ω)) && throw(ArgumentError("At least one element of ω is negative"))
        return new{typeof(h),typeof(α),typeof(γ),typeof(ω)}(h, α, γ, ω)
    end
end

@functor SpectralMixtureKernel

function SpectralMixtureKernel(
    h::Kernel,
    α::AbstractVector{<:Real},
    γ::AbstractMatrix{<:Real},
    ω::AbstractMatrix{<:Real},
)
    size(γ) == size(ω) || throw(DimensionMismatch("γ and ω have different dimensions"))
    return SpectralMixtureKernel(h, α, ColVecs(γ), ColVecs(ω))
end

function SpectralMixtureKernel(
    αs::AbstractVector{<:Real}, γs::AbstractVecOrMat, ωs::AbstractVecOrMat
)
    return SpectralMixtureKernel(SqExponentialKernel(), αs, γs, ωs)
end

function (κ::SpectralMixtureKernel)(x, y)
    xy = x - y
    # use pairwise summation (https://github.com/JuliaLang/julia/pull/31020)
    broadcasted = Broadcast.broadcasted(κ.α, κ.γ, κ.ω) do α, γ, ω
        k = TransformedKernel(κ.kernel, ARDTransform(γ))
        return α * k(x, y) * cospi(2 * dot(ω, xy))
    end
    return sum(Broadcast.instantiate(broadcasted))
end

function Base.show(io::IO, κ::SpectralMixtureKernel)
    return print(
        io,
        "SpectralMixtureKernel Kernel (kernel = ",
        κ.kernel,
        ", # components = ",
        length(κ.α),
        ")",
    )
end

@doc raw"""
    spectral_mixture_product_kernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractMatrix{<:Real},
        γ::AbstractMatrix{<:Real},
        ω::AbstractMatrix{<:Real},
    )

The spectral mixture product is tensor product of spectral mixture kernel applied
on each dimension as described in [1] in equations 13 and 14.
With enough components, the SMP kernel
can model any product kernel to arbitrary precision, and is flexible even
with a small number of components.

## Definition

For inputs ``x, x′ \in \mathbb{R}^D``, the spectral mixture product kernel ``\tilde{k}`` with ``K`` mixture components, mixture weights ``\alpha_1, \alpha_2, \ldots, \alpha_K \in \mathbb{R}^D``, linear transformations ``\gamma_1, \ldots, \gamma_K \in \mathbb{R}^D``, and frequencies ``\omega_1, \ldots, \omega_K \in \mathbb{R}^D`` derived from a translation-invariant kernel ``k`` is defined as

```math
   \tilde{k}(x, x'; \alpha_1, \ldots, \alpha_k, \gamma_1, \ldots, \gamma_K, \omega_1, \ldots, \omega_K, k) = \prod_{i=1}^D \sum_{k=1}^K \alpha_{ik} \cdot h(\gamma_{ik} \cdot x_i, \gamma_{ik} \cdot y_i)) \cdot \cos(2\pi \cdot \omega_{ik} \cdot (x_i - y_i))))
```

## Arguments
- `h`: Stationary kernel (translation invariant), [`SqExponentialKernel`](@ref) by default
- `α`: Weight of each mixture component for each dimension
- `γ`: Linear transformation of the input for `h`.
- `ω`: Frequencies for the cosine function.

`α`, `γ` and `ω` can be an
- `AbstractMatrix` of dimension `D x K` where `D` is the dimension of the inputs 
and `K` is the number of components
- `AbstractVector` of `D` `K`-dimensional `AbstractVector`


# References:
    [1] GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham
"""
function spectral_mixture_product_kernel(
    h::Kernel,
    α::AbstractMatrix{<:Real},
    γ::AbstractMatrix{<:Real},
    ω::AbstractMatrix{<:Real},
)
    (size(α) == size(γ) == size(ω)) ||
        throw(DimensionMismatch("α, γ and ω have different dimensions"))
    return spectral_mixture_product_kernel(h, RowVecs(α), RowVecs(γ), RowVecs(ω))
end

function spectral_mixture_product_kernel(
    h::Kernel,
    α::AbstractVector{<:AbstractVector{<:Real}},
    γ::AbstractVector{<:AbstractVector{<:Real}},
    ω::AbstractVector{<:AbstractVector{<:Real}},
)
    return mapreduce(⊗, α, γ, ω) do αᵢ, γᵢ, ωᵢ
        return SpectralMixtureKernel(h, αᵢ, permutedims(γᵢ), permutedims(ωᵢ))
    end
end

function spectral_mixture_product_kernel(
    α::AbstractVecOrMat, γ::AbstractVecOrMat, ω::AbstractVecOrMat
)
    return spectral_mixture_product_kernel(SqExponentialKernel(), α, γ, ω)
end
