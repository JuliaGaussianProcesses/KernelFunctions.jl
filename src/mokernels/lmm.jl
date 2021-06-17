@doc raw"""
    NaiveLMMMOKernel(g, e::MOKernel, A::AbstractMatrix)

Kernel associated with the linear mixing model.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel is defined as[^BPTHST]
```math
k\big((x, p_x), (x, p_{x'})\big) = H_{p_{x}}K(x, x')H_{p_{x'}}
```
where ``K(x, x') = Diag(k_1(x, x'), ..., k_m(x, x'))`` with zero off-diagonal entries.
``k_1, \ldots, k_m`` are ``m`` kernels, one for each latent process, ``H`` is a
mixing matrix of ``m`` basis vectors spanning the output space.

[^BPTHST]: Wessel P. Bruinsma, Eric Perim, Will Tebbutt, J. Scott Hosking, Arno Solin, Richard E. Turner (2020). [Scalable Exact Inference in Multi-Output Gaussian Processes](https://arxiv.org/pdf/1911.06287.pdf).
"""
struct NaiveLMMMOKernel{Tk<:AbstractVector{<:Kernel}, Th<:AbstractMatrix} <: MOKernel
    K::Tk
    H::Th
    # if just a simple kernel is provided, we construct the MO kernel
    function NaiveLMMMOKernel(k::SimpleKernel, H::AbstractMatrix)
        m = size(H,2)
        K = fill(k, m)
        return new{typeof(K),typeof(H)}(K,H)
    end
    function NaiveLMMMOKernel(K::AbstractVector{<:Kernel}, H::AbstractMatrix)
        return new{typeof(K),typeof(H)}(K,H)
    end
end

function (κ::NaiveLMMMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    (px > size(κ.H, 1) || py > size(κ.H, 1) || px < 1 || py < 1) &&
    error("`px` and `py` must be within the range of the number of outputs")
    m = size(κ.H,2)
    K = Diagonal(map(k -> k(x, y), κ.K))
    return dot(H[px,:], K, H[py,:])
end

# not optimized at all, to be improved
function kernelmatrix(k::NaiveLMMMOKernel, x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
    x.out_dim == size(k.H, 1) ||
        error("Mixing matrix not compatible with the given multi-output inputs")
    m = size(k.H, 2)
    n_x = size(x.x, 1)
    n_y = size(y.x, 1)
    Σ = Matrix(undef, n_x, n_y)
    for i in 1:n_x
        for j in i:n_y
            K_xy = Matrix(undef, m, m)
            K_xy .= 0
            if x.x isa Union{ColVecs,RowVecs}
                x′ = x.x.X[i]
            else
                x′ = x.x[i]
            end
            if y.x isa Union{ColVecs,RowVecs}
                y′ = y.x.X[j]
            else
                y′ = y.x[j]
            end
                for h in 1:m
                    K_xy[h, h] = k.K[h](x′, y′)
                end
            Σ[i,j] = k.H * K_xy * k.H'
            Σ[j,i] = Σ[i,j]
        end
    end
    return Σ
end

function Base.show(io::IO, k::NaiveLMMMOKernel)
    return print(io, "Linear Mixing Model Multi-Output Kernel (naive implementation)")
end

function Base.show(io::IO, ::MIME"text/plain", k::NaiveLMMMOKernel)
    print(io, "Linear Mixing Model Multi-Output Kernel (naive implementation)\n\tkernels (K): ")
    return join(io, k.K, "\n\t\t")
end
