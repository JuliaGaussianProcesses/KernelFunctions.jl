@doc raw"""
    NaiveLMMMOKernel(g, e::MOKernel, A::AbstractMatrix)

Kernel associated with the linear mixing model.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel is defined as[^BPTHST]
```math
k\big((x, p_x), (x, p_{x'})\big) = H_{p_{x}}k_{simple}(x, x')H_{p_{x'}}
```
where ``k_1, \ldots, k_m`` are ``m`` kernels, one for each latent process, ``H`` is a
mixing matrix of ``m`` basis vectors spanning the output space, and ``σ²`` is noise.

[^BPTHST]: Wessel P. Bruinsma, Eric Perim, Will Tebbutt, J. Scott Hosking, Arno Solin, Richard E. Turner (2020). [Scalable Exact Inference in Multi-Output Gaussian Processes](https://arxiv.org/pdf/1911.06287.pdf).
"""
struct NaiveLMMMOKernel{Tk<:Union{Kernel, Vector{<:Kernel}}, Th<:AbstractMatrix, F<:Float64} <: MOKernel
    K::Tk
    H::Th
    σ²::F
    # if just a simple kernel is provided, we construct the MO kernel
    function NaiveLMMMOKernel(k::SimpleKernel, H::AbstractMatrix, σ²)
        σ² >= 0 || error("`σ²` must be positive")
        m = size(H,2)
        K = [k for i in 1:m]
        return new{typeof(K),typeof(H),typeof(σ²)}(K,H,σ²)
    end
    function NaiveLMMMOKernel(K::Vector{<:Kernel}, H::AbstractMatrix, σ²)
        σ² >= 0 || error("`σ²` must be positive")
        return new{typeof(K),typeof(H),typeof(σ²)}(K,H,σ²)
    end
end

function (κ::NaiveLMMMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    (px > size(κ.H, 1) || py > size(κ.H, 1) || px < 1 || py < 1) &&
    error("`px` and `py` must be within the range of the number of outputs")
    m = size(κ.H,2)
    K = Matrix(undef, m, m)
    K .= 0
    for i in 1:m
        K[i, i] = κ.K[i](x, y)
    end
    return H[px,:]' * K * H[py,:]
end

# not optimized at all
function kernelmatrix(k::NaiveLMMMOKernel, x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || error("`x` and `y` should have the same output dimension")
    x.out_dim == size(k.H, 1) ||
        error("Mixing matrix not compatible with the given multi-output inputs")
    m = size(k.H,2)
    n_x = size(x.x,1)
    n_y = size(y.x,1)
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
    return Σ + k.σ²*I
end

function Base.show(io::IO, k::NaiveLMMMOKernel)
    return print(io, "Linear Mixing Model Multi-Output Kernel (naive implementation)")
end

function Base.show(io::IO, ::MIME"text/plain", k::NaiveLMMMOKernel)
    print(io, "Linear Mixing Model Multi-Output Kernel (naive implementation)\n\tkernels (K): ")
    return join(io, k.K, "\n\t\t")
end
