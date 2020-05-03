"""
    SpectralMixtureKernel(
        h::Kernel,
        αs::AbstractVector{<:Real},
        γs::AbstractMatrix{<:Real},
        ωs::AbstractMatrix{<:Real},
    )

Generalised Spectral Mixture kernel function. This family of functions is  dense
in the family of stationary real-valued kernels with respect to the pointwise convergence.[1]

```math
   κ(x, y) = αs' (h(-(V' * t)^2) .* cos(π * M' * t), t = x - y
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
function SpectralMixtureKernel(
    h::Kernel,
    αs::AbstractVector{<:Real},
    γs::AbstractMatrix{<:Real},
    ωs::AbstractMatrix{<:Real},
)
    @assert size(αs, 1) == size(γs, 2) == size(ωs, 2) "The dimensions of αs, γs,
ans ωs do not match"
    @assert size(γs) == size(ωs) "The dimensions of γs ans ωs do not match"

    return sum(zip(αs, eachcol(γs), eachcol(ωs))) do (α, γ, ω)
        a = TransformedKernel(h, LinearTransform(γ'))
        b = TransformedKernel(CosineKernel(), LinearTransform(ω'))
        return α * a * b
    end
end


"""
    SpectralMixtureProductKernel(
        h::Kernel,
        αs::AbstractMatrix{<:Real},
        γs::AbstractMatrix{<:Real},
        ωs::AbstractMatrix{<:Real},
    )

Spectral Mixture Product Kernel.

```math
   κ(x, y) = Πᵢ₌₁ᴷ wᵢᵀ (exp(-½ * vᵢ * t²ᵢ) .* cos(mᵢ * tᵢ)), tᵢ = 2 * π * (xᵢ - yᵢ)
```

# References:
    [1] GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham
"""
function SpectralMixtureProductKernel(
    h::Kernel,
    αs::AbstractMatrix{<:Real},
    γs::AbstractMatrix{<:Real},
    ωs::AbstractMatrix{<:Real},
)
    @assert size(αs) == size(γs) == size(ωs) "The dimensions of αs, γs,
ans ωs do not match"

    return prod(zip(eachrow(αs), eachrow(γs), eachrow(ωs))) do (α, γ, ω)
        return SpectralMixtureKernel(h, α, reshape(γ, 1, :), reshape(ω, 1, :))
    end
end

