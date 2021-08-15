@doc raw"""
    LatentFactorMOKernel(g::AbstractVector{<:Kernel}, e::MOKernel, A::AbstractMatrix)

Kernel associated with the semiparametric latent factor model.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel is defined as[^STJ]
```math
k\big((x, p_x), (x, p_{x'})\big) = \sum^{Q}_{q=1} A_{p_xq}g_q(x, x')A_{p_{x'}q}
                                   + e\big((x, p_x), (x', p_{x'})\big),
```
where ``g_1, \ldots, g_Q`` are ``Q`` kernels, one for each latent process, ``e`` is a
multi-output kernel for ``m`` outputs, and ``A`` is a matrix of weights for the kernels of
size ``m \times Q``.

[^STJ]: M. Seeger, Y. Teh, & M. I. Jordan (2005). [Semiparametric Latent Factor Models](https://infoscience.epfl.ch/record/161465/files/slfm-long.pdf).
"""
struct LatentFactorMOKernel{Tg,Te<:MOKernel,TA<:AbstractMatrix} <: MOKernel
    g::Tg
    e::Te
    A::TA
    function LatentFactorMOKernel(g, e::MOKernel, A::AbstractMatrix)
        all(gi isa Kernel for gi in g) || error("`g` should be an collection of kernels")
        length(g) == size(A, 2) ||
            error("Size of `A` not compatible with the given array of kernels `g`")
        return new{typeof(g),typeof(e),typeof(A)}(g, e, A)
    end
end

function (κ::LatentFactorMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    cov_f = sum(κ.A[px, q] * κ.g[q](x, y) * κ.A[py, q] for q in 1:length(κ.g))
    return cov_f + κ.e((x, px), (y, py))
end

function kernelmatrix(k::LatentFactorMOKernel, x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
    x.out_dim == size(k.A, 1) ||
        error("Kernel not compatible with the given multi-output inputs")

    # Weights matrix ((out_dim x out_dim) x length(k.g))
    W = [col * col' for col in eachcol(k.A)]

    # Latent kernel matrix ((N x N) x length(k.g))
    H = [gi.(x.x, permutedims(y.x)) for gi in k.g]

    # Weighted latent kernel matrix ((N*out_dim) x (N*out_dim))
    W_H = sum(kron(Wi, Hi) for (Wi, Hi) in zip(W, H))

    return W_H .+ kernelmatrix(k.e, x, y)
end

function Base.show(io::IO, k::LatentFactorMOKernel)
    return print(io, "Semi-parametric Latent Factor Multi-Output Kernel")
end

function Base.show(io::IO, ::MIME"text/plain", k::LatentFactorMOKernel)
    print(io, "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: ")
    join(io, k.g, "\n\t\t")
    print(io, "\n\teᵢ: ")
    return join(io, k.e, "\n\t\t")
end
