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

@functor IntrinsicCoregionMOKernel (kernel,)

function ParameterHandling.flatten(::Type{T}, k::IntrinsicCoregionMOKernel) where {T<:Real}
    kernel_vec, unflatten_to_kernel = flatten(T, k.kernel)
    B_vec, unflatten_to_B = value_flatten(T, positive_definite(k.B))
    nkernel = length(kernel_vec)
    ntotal = nkernel + length(B_vec)
    function unflatten_to_intrinsiccoregionkernel(v::Vector{T})
        length(v) == ntotal || error("incorrect number of parameters")
        kernel = unflatten_to_kernel(v[1:nkernel])
        B = unflatten_to_B(v[(nkernel + 1):end])
        return IntrinsicCoregionMOKernel(kernel, B)
    end
    return vcat(kernel_vec, B_vec), unflatten_to_intrinsiccoregionkernel
end

function (k::IntrinsicCoregionMOKernel)((x, px)::Tuple{Any,Int}, (y, py)::Tuple{Any,Int})
    return k.B[px, py] * k.kernel(x, y)
end

function _mo_output_covariance(k::IntrinsicCoregionMOKernel, out_dim)
    @assert size(k.B) == (out_dim, out_dim)
    return k.B
end

function kernelmatrix(
    k::IntrinsicCoregionMOKernel, x::MOI, y::MOI
) where {MOI<:IsotopicMOInputsUnion}
    x.out_dim == y.out_dim ||
        throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
    Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
    Koutputs = _mo_output_covariance(k, x.out_dim)
    return _kernelmatrix_kron_helper(MOI, Kfeatures, Koutputs)
end

if VERSION >= v"1.6"
    function kernelmatrix!(
        K::AbstractMatrix, k::IntrinsicCoregionMOKernel, x::MOI, y::MOI
    ) where {MOI<:IsotopicMOInputsUnion}
        x.out_dim == y.out_dim ||
            throw(DimensionMismatch("`x` and `y` must have the same `out_dim`"))
        Kfeatures = kernelmatrix(k.kernel, x.x, y.x)
        Koutputs = _mo_output_covariance(k, x.out_dim)
        return _kernelmatrix_kron_helper!(K, MOI, Kfeatures, Koutputs)
    end
end

function Base.show(io::IO, k::IntrinsicCoregionMOKernel)
    return print(
        io, "Intrinsic Coregion Kernel: ", k.kernel, " with ", size(k.B, 1), " outputs"
    )
end
