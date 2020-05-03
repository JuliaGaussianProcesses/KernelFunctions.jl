"""
    SpectralMixtureKernel(αs<:AbstractVector{<:Real}, γs<:AbstractMatrix{<:Real}, ωs<:AbstractMatrix{<:Real})

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
    γs::AbstractVector{<:AbstractVector{<:Real}},
    ωs::AbstractVector{<:AbstractVector{<:Real}},
)
    return sum(zip(αs, γs, ωs)) do (α, γ, ω)
        a = TransformedKernel(h, LinearTransform(γ'))
        b = TransformedKernel(CosineKernel(), LinearTransform(ω'))
        return α * a * b
    end
end


"""
    SpectralMixtureProductKernel(W<:AbstractMatrix{<:Real}, M<:AbstractMatrix{<:Real}, V<:AbstractMatrix{<:Real})

Spectral Mixture Product Kernel.

```math
   κ(x, y) = Πᵢ₌₁ᴷ w^tᵢ (exp(-½ * vᵢ * t²ᵢ) .* cos(mᵢ * tᵢ)), tᵢ = 2 * π * (xᵢ - yᵢ)
```

# References:
    [1] GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham
"""
# struct SpectralMixtureProductKernel{
#     M1<:AbstractMatrix{<:Real},
#     M2<:AbstractMatrix{<:Real},
#     M3<:AbstractMatrix{<:Real}} <: BaseKernel
#     W::M1
#     M::M2
#     V::M3
#     function SpectralMixtureProductKernel(;W , M , V)
#         @assert size(W) == size(M) == size(V) "Dimensions of weights W, means M and variances V do not match."
#         new{typeof(W),typeof(M),typeof(V)}(W, M, V)
#     end
# end

# function (κ::SpectralMixtureProductKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
#     t = 2π * (x - y)
#     @info zip(κ.W, κ.M, κ.V, t)
#     return 
#     #return dot(κ.w, (cos.((t.^2)' * κ.V) .* exp.(t' * κ.M))')
# end

# Base.show(io::IO, κ::SpectralMixtureProductKernel) =
#     print(io, "Spectral Mixture Product Kernel (with D=", size(κ.M, 1), ", Q=",
#           size(κ.M, 2), ")")

