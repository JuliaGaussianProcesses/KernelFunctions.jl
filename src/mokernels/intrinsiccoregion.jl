@doc raw"""
    IntrinsicCoregionMOKernel(; kernel::Kernel, B::AbstractMatrix)

Kernel associated with the intrinsic coregionalization model.

# Definition

For inputs ``x, x'`` and output dimensions ``p, p'``, the kernel is defined as[^ARL]
```math
k\big((x, p), (x', p'); B, \tilde{k}\big) = B_{p, p'} \tilde{k}\big(x, x'\big),
```
where ``B`` is a positive semidefinite matrix of size ``m \times m``, with ``m`` being the
number of outputs, and ``\tilde{k}`` is a scalar-valued kernel shared by the latent
processes.

[^ARL]: M. Álvarez, L. Rosasco, & N. Lawrence (2012). [Kernels for Vector-Valued Functions: a Review](https://arxiv.org/pdf/1106.6251.pdf).
"""
struct IntrinsicCoregionMOKernel{K<:Kernel,T<:AbstractMatrix} <: MOKernel
    kernel::K
    B::T

    function IntrinsicCoregionMOKernel{K,T}(kernel::K, B::T) where {K,T}
        @check_args(
            IntrinsicCoregionMOKernel,
            B,
            eigmin(B) >= 0,
            "B has to be positive semi-definite"
        )
        return new{K,T}(kernel, B)
    end
end

function IntrinsicCoregionMOKernel(; kernel::Kernel, B::AbstractMatrix)
    return IntrinsicCoregionMOKernel{typeof(kernel),typeof(B)}(kernel, B)
end

function IntrinsicCoregionMOKernel(kernel::Kernel, B::AbstractMatrix)
    return IntrinsicCoregionMOKernel{typeof(kernel),typeof(B)}(kernel, B)
end

function (k::IntrinsicCoregionMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int}) 
    return k.B[px, py] * k.kernel(x, y)
end

# convenience function
export matrixkernel
function matrixkernel(k::IntrinsicCoregionMOKernel, x::Vector{T}, y::Vector{T}) where T <: Real
    @assert size(x) == size(y)
    outputsize = size(k.B, 1)
    xMO = MOInputIsotopicByFeatures([x], outputsize)
    yMO = MOInputIsotopicByFeatures([y], outputsize)
    kernelmatrix(k, xMO, yMO)
end

function kernelmatrix(k::IntrinsicCoregionMOKernel, x::MOInputIsotopicByFeatures, y::MOInputIsotopicByFeatures)
    @assert x.out_dim == y.out_dim == size(k.B, 1)
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    kron(Ktmp, k.B)
end

function kernelmatrix!(K::AbstractMatrix, k::IntrinsicCoregionMOKernel, x::MOInputIsotopicByFeatures, y::MOInputIsotopicByFeatures)
    @assert x.out_dim == y.out_dim == size(k.B, 1)
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    kron!(K, Ktmp, k.B)
end


function kernelmatrix(k::IntrinsicCoregionMOKernel, x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs)
    @assert x.out_dim == y.out_dim == size(k.B, 1)
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    kron(k.B, Ktmp)
end

function kernelmatrix!(K::AbstractMatrix, k::IntrinsicCoregionMOKernel, x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs)
    @assert x.out_dim == y.out_dim == size(k.B, 1)
    Ktmp = kernelmatrix(k.kernel, x.x, y.x)
    kron!(K, k.B, Ktmp)
end


function Base.show(io::IO, k::IntrinsicCoregionMOKernel)
    return print(
        io, "Intrinsic Coregion Kernel: ", k.kernel, " with ", size(k.B, 1), " outputs"
    )
end
