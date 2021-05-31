@doc raw"""
    IntrinsicIntrinsicCoregionMOKernel(kernel::Kernel, B::AbstractMatrix)

Kernel associated with the intrinsic coregionalization model.

# Definition

For inputs ``x, x'`` and output dimensions ``p_x, p_{x'}'``, the kernel is defined as[^ARL]
```math
k\big((x, p_x), (x', p_{x'}); B, \tilde{k}\big) = [B]_{p_x, p_{x'}} \tilde{k}\big(x, x'\big),
```
where ``B`` is a positive semidefinite matrix of size ``m \times m``, with ``m`` being the number of outputs, and ``\tilde{k}`` is a scalar-valued kernel shared by the latent processes.

[^ARL]: M. Álvarez, L. Rosasco, & N. Lawrence (2012). [Kernels for Vector-Valued Functions: a Review](https://arxiv.org/pdf/1106.6251.pdf).
"""
struct IntrinsicCoregionMOKernel{K<:Kernel,T<:AbstractMatrix} <: MOKernel
    kernel::K
    B::T

    function IntrinsicCoregionMOKernel{K,T}(kernel::K, B::T) where {K,T}
        @check_args(IntrinsicCoregionMOKernel, B, (eigmin(B) >= 0), "B is Positive semi-definite")
        return new{K,T}(kernel, B)
    end
end

function IntrinsicCoregionMOKernel(kernel::Kernel, B::AbstractMatrix)
    return IntrinsicCoregionMOKernel{typeof(kernel),typeof(B)}(kernel, B)
end

function (k::IntrinsicCoregionMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return k.B[px, py] * k.kernel(x, y)
end

function Base.show(io::IO, ::IntrinsicCoregionMOKernel)
    return print(io, "Intrinsic Coregion Multi-Output Kernel")
end