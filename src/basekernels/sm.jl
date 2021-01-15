"""
    spectral_mixture_kernel(
        h::Kernel=SqExponentialKernel(),
        αs::AbstractVector{<:Real},
        γs::AbstractMatrix{<:Real},
        ωs::AbstractMatrix{<:Real},
    )

where αs are the weights of dimension (A, ), γs is the covariance matrix of
dimension (D, A) and ωs are the mean vectors and is of dimension (D, A).
Here, D is input dimension and A is the number of spectral components.

`h` is the kernel, which defaults to [`SqExponentialKernel`](@ref) if not specified.

Generalised Spectral Mixture kernel function. This family of functions is  dense
in the family of stationary real-valued kernels with respect to the pointwise convergence.[1]

```math
   κ(x, y) = αs' (h(-(γs' * t)^2) .* cos(π * ωs' * t), t = x - y
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
    αs::AbstractVector{<:Real},
    γs::AbstractMatrix{<:Real},
    ωs::AbstractMatrix{<:Real},
)
    if !(size(αs, 1) == size(γs, 2) == size(ωs, 2))
        throw(DimensionMismatch("The dimensions of αs, γs, ans ωs do not match"))
    end
    if size(γs) != size(ωs)
        throw(DimensionMismatch("The dimensions of γs ans ωs do not match"))
    end

    return sum(zip(αs, eachcol(γs), eachcol(ωs))) do (α, γ, ω)
        a = TransformedKernel(h, LinearTransform(γ'))
        b = TransformedKernel(CosineKernel(), LinearTransform(ω'))
        return α * a * b
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
    return TensorProduct(
        spectral_mixture_kernel(h, α, reshape(γ, 1, :), reshape(ω, 1, :)) for
        (α, γ, ω) in zip(eachrow(αs), eachrow(γs), eachrow(ωs))
    )
end

function spectral_mixture_product_kernel(
    αs::AbstractMatrix{<:Real}, γs::AbstractMatrix{<:Real}, ωs::AbstractMatrix{<:Real}
)
    return spectral_mixture_product_kernel(SqExponentialKernel(), αs, γs, ωs)
end
