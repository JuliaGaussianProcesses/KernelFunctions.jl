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
        γ::AbstractVector{<:AbstractVecorMat{<:Real}},
        ω::AbstractVector{<:AbstractVecorMat{<:Real}},
    )

Given `A` the number of mixture components and `D` the dimension of the inputs:

## Arguments
- `h`: Stationary kernel (translation invariant), `SqExponentialKernel` by default
- `α`: Weight vector of each mixture component (`length(α)==A`)
- `γ`: Linear transformation of the input for `h`. `γ` can be an `AbstractMatrix` or 
- `ω`: Linear transformation 
where `α` are the weight vector of dimension `A`, `γs` is the sqrt of the covariance matrix of
dimension `(D, A)` and `ωs` are the concatenated mean vectors of dimension (D, A).
Here, `D` is input dimension and `A` is the number of spectral components.

`h` is the stationary kernel, which defaults to [`SqExponentialKernel`](@ref) if not specified.

Generalised Spectral Mixture kernel function. This family of functions is dense
in the family of stationary real-valued kernels with respect to the pointwise convergence.[1]

```math
   κ(x, y) = \sum_k \alpha_k^\top (h(γ_k \odot x, γ_k \odot y) \cos(π \cdot ω_k^\top (x-y)),
```

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
    if !(size(α, 1) == size(γ, 2) == size(ω, 2))
        throw(DimensionMismatch("The dimensions of α, γ, ans ω do not match"))
    end
    if size(γ, 1) != size(ω, 1)
        throw(DimensionMismatch("The dimensions of γ ans ω do not match"))
    end

    return sum(zip(α, eachcol(γ), eachcol(ω))) do (α, γ, ω)
        a = TransformedKernel(h, ARDTransform(γ))
        b = TransformedKernel(CosineKernel(), ARDTransform(ω))
        return α * a * b
    end
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

    return sum(zip(α, γ, ω)) do (αk, γk, ωk)
        a = TransformedKernel(h, ARDTransform(γk))
        b = TransformedKernel(CosineKernel(), ARDTransform(ωk))
        return αk * a * b
    end
end


function spectral_mixture_kernel(
    αs::AbstractVector{<:Real}, γs::AbstractMatrix{<:Real}, ωs::AbstractMatrix{<:Real}
)
    return spectral_mixture_kernel(SqExponentialKernel(), αs, γs, ωs)
end

"""
    spectral_mixture_product_kernel(
        h::Kernel=SqExponentialKernel(),
        αs::AbstractMatrix{<:Real},
        γs::AbstractMatrix{<:Real},
        ωs::AbstractMatrix{<:Real},
    )

where αs are the weights of dimension (D, A), γs is the covariance matrix of
dimension (D, A) and ωs are the mean vectors and is of dimension (D, A).
Here, D is input dimension and A is the number of spectral components.

Spectral Mixture Product Kernel. With enough components A, the SMP kernel
can model any product kernel to arbitrary precision, and is flexible even
with a small number of components [1]


`h` is the kernel, which defaults to [`SqExponentialKernel`](@ref) if not specified.

```math
   κ(x, y) = Πᵢ₌₁ᴷ Σ(αsᵢᵀ .* (h(-(γsᵢᵀ * tᵢ)²) .* cos(ωsᵢᵀ * tᵢ))), tᵢ = xᵢ - yᵢ
```

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
    αs::AbstractMatrix{<:Real}, γs::AbstractMatrix{<:Real}, ωs::AbstractMatrix{<:Real}
)
    return spectral_mixture_product_kernel(SqExponentialKernel(), αs, γs, ωs)
end
