@doc raw"""
    LinearMixingModelKernel(k::Kernel, H::AbstractMatrix)
    LinearMixingModelKernel(Tk::AbstractVector{<:Kernel},Th::AbstractMatrix)

Kernel associated with the linear mixing model, taking a vector of `m` kernels and a `m × p` matrix H for a function with `p` outputs. Also accepts a single kernel `k` for use across all `m` basis vectors. 

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel is defined as[^BPTHST]
```math
k\big((x, p_x), (x, p_{x'})\big) = H_{:,p_{x}}K(x, x')H_{:,p_{x'}}
```
where ``K(x, x') = Diag(k_1(x, x'), ..., k_m(x, x'))`` with zero off-diagonal entries.
``H_{:,p_{x}}`` is the ``p_x``-th column (`p_x`-th output) of ``H \in \mathbb{R}^{m \times p}``
representing ``m`` basis vectors for the ``p`` dimensional output space of ``f``.
``k_1, \ldots, k_m`` are ``m`` kernels, one for each latent process, ``H`` is a
mixing matrix of ``m`` basis vectors spanning the output space.

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

## current implementation
# julia> @benchmark KernelFunctions.kernelmatrix(k, x1IO, x2IO)
# BenchmarkTools.Trial: 3478 samples with 1 evaluation.
#  Range (min … max):  1.362 ms …   5.498 ms  ┊ GC (min … max): 0.00% … 72.47%
#  Time  (median):     1.396 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.435 ms ± 358.577 μs  ┊ GC (mean ± σ):  2.28% ±  6.70%

#   ▂▆█▇▄▂  ▂▁                                                  ▁
#   ███████▆██▅▅▁▄▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ █
#   1.36 ms      Histogram: log(frequency) by time       2.1 ms <

#  Memory estimate: 410.73 KiB, allocs estimate: 23090.

## proposed improvement
# julia> @benchmark KernelFunctions.kernelmatrix2(k, x1IO, x2IO)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  16.871 μs …   3.440 ms  ┊ GC (min … max):  0.00% … 97.80%
#  Time  (median):     18.625 μs               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   24.734 μs ± 129.308 μs  ┊ GC (mean ± σ):  20.63% ±  3.92%

#    ▄▆███▇▆▅▄▄▂▂                           ▁ ▁                  ▂
#   ████████████████▇▆▄▅▅▄▅▅▅▅▅▆▃▄▅▃▂▂▃▃▄▆▇▇████████▅▆▅▃▅▄▆▅▆▆▆▇ █
#   16.9 μs       Histogram: log(frequency) by time      36.4 μs <

#  Memory estimate: 84.56 KiB, allocs estimate: 338.

function kernelmatrix2(k::LinearMixingModelKernel, X, Y)
    K = [kernelmatrix(ki, X.x, Y.x) for ki in k.K]
    L = size(k.H, 2)
    return reduce(hcat, [reduce(vcat, [sum(k.H[:,i].*(K .* k.H[:,j])) for i in 1:L]) for j in 1:L])
end

# function matrixkernel(k::LinearMixingModelKernel, x, y)
#     return matrixkernel(k, x, y, size(k.H, 2))
# end

function matrixkernel(k::LinearMixingModelKernel, x, y)
    K = [ki(x, y) for ki in k.K]
    return k.H' * ( K .* k.H)
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
