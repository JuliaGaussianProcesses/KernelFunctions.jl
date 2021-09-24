@doc raw"""
    LinearMixingModelKernel(k::Kernel, H::AbstractMatrix)
    LinearMixingModelKernel(Tk::AbstractVector{<:Kernel},Th::AbstractMatrix)

Kernel associated with the linear mixing model, taking a vector of `Q` kernels and a `Q × m` mixing matrix H for a function with `m` outputs. Also accepts a single kernel `k` for use across all `Q` basis vectors. 

# Definition

For inputs ``x, x'`` and output dimensions ``p, p''``, the kernel is defined as[^BPTHST]
```math
k\big((x, p), (x, p')\big) = H_{:,p}K(x, x')H_{:,p'}
```
where ``K(x, x') = Diag(k_1(x, x'), ..., k_Q(x, x'))`` with zero off-diagonal entries.
``H_{:,p}`` is the ``p``-th column (`p`-th output) of ``H \in \mathbb{R}^{Q \times m}``
representing ``Q`` basis vectors for the ``m`` dimensional output space of ``f``.
``k_1, \ldots, k_Q`` are ``Q`` kernels, one for each latent process, ``H`` is a
mixing matrix of ``Q`` basis vectors spanning the output space.

[^BPTHST]: Wessel P. Bruinsma, Eric Perim, Will Tebbutt, J. Scott Hosking, Arno Solin, Richard E. Turner (2020). [Scalable Exact Inference in Multi-Output Gaussian Processes](https://arxiv.org/pdf/1911.06287.pdf).
"""
struct LinearMixingModelKernel{Tk<:AbstractVector{<:Kernel},Th<:AbstractMatrix} <: MOKernel
    K::Tk
    H::Th
    function LinearMixingModelKernel(Tk::AbstractVector{<:Kernel}, H::AbstractMatrix)
        @assert length(Tk) == size(H, 1) "Number of kernels and number of rows in H must match"
        return new{typeof(Tk),typeof(H)}(Tk, H)
    end
end

function LinearMixingModelKernel(k::Kernel, H::AbstractMatrix)
    return LinearMixingModelKernel(Fill(k, size(H, 1)), H)
end

function (κ::LinearMixingModelKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    (px > size(κ.H, 2) || py > size(κ.H, 2) || px < 1 || py < 1) &&
        error("`px` and `py` must be within the range of the number of outputs")
    return sum(κ.H[i, px] * κ.K[i](x, y) * κ.H[i, py] for i in 1:length(κ.K))
end

function Base.show(io::IO, k::LinearMixingModelKernel)
    return print(io, "Linear Mixing Model Multi-Output Kernel")
end

function Base.show(io::IO, mime::MIME"text/plain", k::LinearMixingModelKernel)
    print(io, "Linear Mixing Model Multi-Output Kernel. Kernels:")
    for k in k.K
        print(io, "\n\t")
        show(io, mime, k)
    end
end
