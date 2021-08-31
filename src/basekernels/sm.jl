"""
    spectral_mixture_kernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractVector{<:Real},
        γ::AbstractMatrix{<:Real},
        ω::AbstractMatrix{<:Real},
    )
    spectral_mixture_kernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractVector{<:Real},
        γ::AbstractVector{<:AbstractVecOrMat{<:Real}},
        ω::AbstractVector{<:AbstractVecOrMat{<:Real}},
    )

Generalised Spectral Mixture kernel function as described in [1] (Eq. 6). This family of functions is dense
in the family of stationary real-valued kernels with respect to the pointwise convergence.[1]

```math
   κ(x, y) = \sum_{k=1}^K \alpha_k (h(γ_k \odot x, γ_k \odot y) \cos(2π \cdot ω_k^\top (x-y)),
```

## Arguments
- `h`: Stationary kernel (translation invariant), [`SqExponentialKernel`](@ref) by default
- `α`: Weight vector of each mixture component
- `γ`: Linear transformation of the input for `h`.
- `ω`: Linear transformation of the input for the [`CosineKernel`](@ref).

`γ` and `ω` can either be
- `AbstractMatrix` of dimension `D x K` where `D` is the dimension of the inputs 
and `K` is the number of components
- `AbstractVector`s (of length `A`) of `AbstractVector` of length `D`


# References:
    [1] Generalized Spectral Kernels, by Yves-Laurent Kom Samo and Stephen J. Roberts
    [2] SM: Gaussian Process Kernels for Pattern Discovery and Extrapolation,
            ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams,
    [3] Covariance kernels for fast automatic pattern discovery and extrapolation
        with Gaussian processes, Andrew Gordon Wilson, PhD Thesis, January 2014.
        http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
    [4] http://www.cs.cmu.edu/~andrewgw/pattern/.

"""
function spectral_mixture_kernel(
    h::Kernel,
    α::AbstractVector{<:Real},
    γ::AbstractMatrix{<:Real},
    ω::AbstractMatrix{<:Real},
)
    return spectral_mixture_kernel(h, α, ColVecs(γ), ColVecs(ω))
end

function spectral_mixture_kernel(
    h::Kernel,
    α::AbstractVector{<:Real},
    γ::AbstractVector{<:AbstractVector},
    ω::AbstractVector{<:AbstractVector},
)
    if !(length(α) == length(γ) == length(ω))
        throw(DimensionMismatch("The dimensions of α, γ, ans ω do not match"))
    end

    return mapreduce(+, α, γ, ω) do (αk, γk, ωk)
        sqkernel = TransformedKernel(h, ARDTransform(γk))
        coskernel = TransformedKernel(CosineKernel(), ARDTransform(2 * ωk))
        return αk * sqkernel * coskernel
    end
end

function spectral_mixture_kernel(
    αs::AbstractVector{<:Real}, γs::AbstractVecOrMat, ωs::AbstractVecOrMat
)
    return spectral_mixture_kernel(SqExponentialKernel(), αs, γs, ωs)
end

"""
    spectral_mixture_product_kernel(
        h::Kernel=SqExponentialKernel(),
        α::AbstractMatrix{<:Real},
        γ::AbstractMatrix{<:Real},
        ω::AbstractMatrix{<:Real},
    )

The spectral mixture product is tensor product of spectral mixture kernel applied
on each dimension as described in [1] (Eq. 13, 14).
With enough components, the SMP kernel
can model any product kernel to arbitrary precision, and is flexible even
with a small number of components

```math
   κ(x, y) = \prod_{i=1}^D \sum_{k=1}^K \alpha_{k,i}  (h(\gamma_{k,i} x_i, \gamma_{k,i} y_i)) cos(2\pi \omega_{i, k} (x_i - y_i))))
```

## Arguments
- `h`: Stationary kernel (translation invariant), [`SqExponentialKernel`](@ref) by default
- `α`: Weight of each mixture component for each dimension
- `γ`: Linear transformation of the input for `h`.
- `ω`: Linear transformation of the input for the [`CosineKernel`](@ref).

`α`, `γ` and `ω` can all be
where αs are the weights of dimension (D, A), γs is the covariance matrix of
dimension (D, A) and ωs are the mean vectors and is of dimension (D, A).
Here, D is input dimension and A is the number of spectral components.

# References:
    [1] GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham
"""
function spectral_mixture_product_kernel(
    h::Kernel,
    αs::AbstractMatrix{<:Real},
    γs::AbstractMatrix{<:Real},
    ωs::AbstractMatrix{<:Real},
)
    if !(size(αs) == size(γs) == size(ωs))
        throw(DimensionMismatch("The dimensions of αs, γs, ans ωs do not match"))
    end
    return KernelTensorProduct(
        spectral_mixture_kernel(h, α, reshape(γ, 1, :), reshape(ω, 1, :)) for
        (α, γ, ω) in zip(eachrow(αs), eachrow(γs), eachrow(ωs))
    )
end

function spectral_mixture_product_kernel(
    αs::AbstractVecOrMat, γs::AbstractVecOrMat, ωs::AbstractVecOrMat
)
    return spectral_mixture_product_kernel(SqExponentialKernel(), αs, γs, ωs)
end
