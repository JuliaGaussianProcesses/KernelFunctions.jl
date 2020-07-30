@doc raw"""
    LatentFactorMOKernel(
        g::AbstractVector{<:Kernel}, 
        e::AbstractVector{<:Kernel}, 
        A::AbstractMatrix
        )

A semiparametric kernel for problems involving multiple response variables.

``k((x, p), (y, p)) = k_p(x, y) = \Sum^{Q}_{q=1} A_{pq}g_q(x, y) + e_p(x, y)``

# Arguments
- `g::AbstractVector{<:Kernel}`: an array of kernels
- `e::AbstractVector{<:Kernel}`: an array of kernels
- `A::AbstractMatrix`: an matrix of weights for the kernels of shape (length(e), length(g))


# Reference:
- [Seeger, Teh, and Jordan (2005)](https://infoscience.epfl.ch/record/161465/files/slfm-long.pdf)

"""
struct LatentFactorMOKernel{
    Tg <:AbstractVector{<:Kernel}, 
    Te <:AbstractVector{<:Kernel}, 
    TA <: AbstractMatrix
    } <: Kernel
    g::Tg
    e::Te
    A::TA
    function LatentFactorMOKernel(
        g::AbstractVector{<:Kernel}, 
        e::AbstractVector{<:Kernel}, 
        A::AbstractMatrix
        )
        @assert (length(e), length(g)) == 
            size(A) "Size of A not compatible to the given array of kernels"
        return new{typeof(g), typeof(e), typeof(A)}(g, e, A)
    end
end

function (κ::LatentFactorMOKernel)(x::Tuple{Vector, Int}, y::Tuple{Vector, Int})
    if last(x) == last(y)
        return sum([κ.g[i](first(x), first(y)) * κ.A[last(x), i] for i in 1:length(κ.g)]) + 
        κ.e[last(x)](first(x), first(y))
    else
        return 0.0
    end
end

function kernelmatrix(k::LatentFactorMOKernel, x::MOInput, y::MOInput)
    @assert x.out_dim == y.out_dim
    @assert x.out_dim == 
        size(k.A, 1) "Kernel not compatible with the given multi-output inputs"
    return k.(x, permutedims(collect(y)))
end

function Base.show(io::IO, k::LatentFactorMOKernel)
    print(
        io, 
        string(
            "Semi-parametric Latent Factor Multi-Output Kernel\n\t gs: ", 
            string(k.g),
            "\n\t es: ",
            string(k.e),
            )
        )
end