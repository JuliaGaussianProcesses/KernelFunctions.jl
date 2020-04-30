"""
    SpectralMixtureKernel(w<:AbstractVector{<:Real}, M<:AbstractMatrix{<:Real}, V<:AbstractMatrix{<:Real})

Gaussian Spectral Mixture kernel function.

```math
   κ(x, y) = w' (exp(- 1 / 2 * V' * t^2) .* cos(M' * t), t = x - y
```

# References:
    [1] SM: Gaussian Process Kernels for Pattern Discovery and Extrapolation,
        ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams,
    [2] SMP: GPatt: Fast Multidimensional Pattern Extrapolation with GPs,
        arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa,
        Arye Nehorai and John P. Cunningham, and
    [3] Covariance kernels for fast automatic pattern discovery and extrapolation
        with Gaussian processes, Andrew Gordon Wilson, PhD Thesis, January 2014.
        http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf
    [4] http://www.cs.cmu.edu/~andrewgw/pattern/.

"""
struct SpectralMixtureKernel{
    V<:AbstractVector{<:Real},
    M1<:AbstractMatrix{<:Real},
    M2<:AbstractMatrix{<:Real}} <: BaseKernel
    w::V
    M::M1
    V::M2
    function SpectralMixtureKernel(;w , M , V)
        @assert size(M) == size(V) "Dimensions of means m and variances v do not match."
        @assert size(w, 1) == size(M, 2) == size(V, 2) "First dimension of weights w, means m, variances v are does not match."
        new{typeof(w),typeof(M),typeof(V)}(w, M, V)
    end
end

function (κ::SpectralMixtureKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    t = 2π * (x - y)
    return dot(κ.w, (cos.((t.^2)' * κ.V) .* exp.(t' * κ.M))')
end

Base.show(io::IO, κ::SpectralMixtureKernel) = print(io, "Spectral Mixture Kernel (with D=",
                                                    size(κ.M, 1), ", Q=", size(κ.M, 2), ")")


"""
    SpectralMixtureProductKernel(W<:AbstractMatrix{<:Real}, M<:AbstractMatrix{<:Real}, V<:AbstractMatrix{<:Real})

Spectral Mixture Product Kernel.

```math
   \kappa(x, y) = \Pi_{d=1}^D w^t_d (exp(-\frac{1}{2} * v_d * t^2_d) .* cos(m_d * t_d)), t_d = 2 * \pi * (x_d - y_d)
```
"""
struct SpectralMixtureProductKernel{
    M1<:AbstractMatrix{<:Real},
    M2<:AbstractMatrix{<:Real},
    M3<:AbstractMatrix{<:Real}} <: BaseKernel
    W::M1
    M::M2
    V::M3
    function SpectralMixtureProductKernel(;W , M , V)
        @assert size(W) == size(M) == size(V) "Dimensions of weights W, means M and variances V do not match."
        new{typeof(W),typeof(M),typeof(V)}(W, M, V)
    end
end

function (κ::SpectralMixtureProductKernel)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    t = 2π * (x - y)
    @info zip(κ.W, κ.M, κ.V, t)
    return 1
    #return dot(κ.w, (cos.((t.^2)' * κ.V) .* exp.(t' * κ.M))')
end

Base.show(io::IO, κ::SpectralMixtureProductKernel) =
    print(io, "Spectral Mixture Product Kernel (with D=", size(κ.M, 1), ", Q=",
          size(κ.M, 2), ")")

