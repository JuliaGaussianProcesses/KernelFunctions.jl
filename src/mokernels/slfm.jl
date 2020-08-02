@doc raw"""
    LatentFactorMOKernel(
        g,
        e,
        A::AbstractMatrix
        )

The kernel associated with the Semiparametric Latent Factor Model, introduced by 
Seeger, Teh and Jordan (2005).

``k((x, p), (y, p)) = k_p(x, y) = \Sum^{Q}_{q=1} A_{pq}g_q(x, y) + e_p(x, y)``

# Arguments
- `g`: a collection of kernels, one for each latent process
- `e`: a collection of kernels, one for each output
- `A::AbstractMatrix`: a matrix of weights for the kernels of size `(length(e), length(g))`


# Reference:
- [Seeger, Teh, and Jordan (2005)](https://infoscience.epfl.ch/record/161465/files/slfm-long.pdf)

"""
struct LatentFactorMOKernel{Tg, Te, TA <: AbstractMatrix} <: Kernel
    g::Tg
    e::Te
    A::TA
    function LatentFactorMOKernel(g, e, A::AbstractMatrix)
        all(gi isa Kernel for gi in g) || error("`g` should be an collection of kernels")
        all(ei isa Kernel for ei in e) || error("`e` should be an collection of kernels")
        (length(e), length(g)) == size(A) || 
            error("Size of A not compatible to the given array of kernels")
        return new{typeof(g), typeof(e), typeof(A)}(g, e, A)
    end
end

function (κ::LatentFactorMOKernel)((x, px)::Tuple{Vector, Int}, (y, py)::Tuple{Vector, Int})
    if px == py
        return sum(κ.g[i](x, y) * κ.A[px, i] for i in 1:length(κ.g)) + 
            κ.e[px](x, y)
    else
        return 0.0
    end
end

function kernelmatrix(k::LatentFactorMOKernel, x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
    x.out_dim == size(k.A, 1) ||
        error("Kernel not compatible with the given multi-output inputs")
    return k.(x, permutedims(collect(y)))
end

function Base.show(io::IO, k::LatentFactorMOKernel)
    print(io, "Semi-parametric Latent Factor Multi-Output Kernel")
end

function Base.show(io::IO, ::MIME"text/plain", k::LatentFactorMOKernel)
    print(io, "Semi-parametric Latent Factor Multi-Output Kernel\n\tgᵢ: ")
    join(io, k.g, "\n\t\t")
    print(io, "\n\teᵢ: ")
    join(io, k.e, "\n\t\t")
end
