"""
    TransformedKernel(k::Kernel, t::Transform)

Kernel derived from `k` for which inputs are transformed via a [`Transform`](@ref) `t`.

It is preferred to create kernels with input transformations with [`∘`](@ref) or its
alias [`compose`](@ref) instead of `TransformedKernel` directly since this allows optimized
implementations for specific kernels and transformations.

# Definition

For inputs ``x, x'``, the transformed kernel ``\\widetilde{k}`` derived from kernel ``k`` by
input transformation ``t`` is defined as
```math
\\widetilde{k}(x, x'; k, t) = k\\big(t(x), t(x')\\big).
```
"""
struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

@functor TransformedKernel

(k::TransformedKernel)(x, y) = k.kernel(k.transform(x), k.transform(y))

# Optimizations for scale transforms of simple kernels to save allocations:
# Instead of a multiplying every element of the inputs before evaluating the metric,
# we perform a scalar multiplcation of the distance of the original inputs, if possible.
function (k::TransformedKernel{<:SimpleKernel,<:ScaleTransform})(
    x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return kappa(k.kernel, _scale(k.transform, metric(k.kernel), x, y))
end

function _scale(t::ScaleTransform, metric::Euclidean, x, y)
    return first(t.s) * evaluate(metric, x, y)
end
function _scale(t::ScaleTransform, metric::Union{SqEuclidean,DotProduct}, x, y)
    return first(t.s)^2 * evaluate(metric, x, y)
end
_scale(t::ScaleTransform, metric, x, y) = evaluate(metric, t(x), t(y))

"""
    kernel ∘ transform
    ∘(kernel, transform)
    compose(kernel, transform)

Compose a `kernel` with a transformation `transform` of its inputs.

If no optimized implementation exists, a [`TransformedKernel`](@ref) is created.

See also: [`TransformedKernel`](@ref)
"""
Base.:∘(k::Kernel, t::Transform) = TransformedKernel(k, t)
Base.:∘(k::TransformedKernel, t::Transform) = TransformedKernel(k.kernel, k.transform ∘ t)

Base.show(io::IO, κ::TransformedKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::TransformedKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    return print(io, "\n" * ("\t"^(shift + 1)) * "- $(κ.transform)")
end

# Kernel matrix operations

function kernelmatrix_diag!(K::AbstractVector, κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix_diag!(K, κ.kernel, _map(κ.transform, x))
end

function kernelmatrix_diag!(
    K::AbstractVector, κ::TransformedKernel, x::AbstractVector, y::AbstractVector
)
    return kernelmatrix_diag!(K, κ.kernel, _map(κ.transform, x), _map(κ.transform, y))
end

function kernelmatrix!(K::AbstractMatrix, κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix!(K, κ.kernel, _map(κ.transform, x))
end

function kernelmatrix!(
    K::AbstractMatrix, κ::TransformedKernel, x::AbstractVector, y::AbstractVector
)
    return kernelmatrix!(K, κ.kernel, _map(κ.transform, x), _map(κ.transform, y))
end

function kernelmatrix_diag(κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix_diag(κ.kernel, _map(κ.transform, x))
end

function kernelmatrix_diag(κ::TransformedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix_diag(κ.kernel, _map(κ.transform, x), _map(κ.transform, y))
end

function kernelmatrix(κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix(κ.kernel, _map(κ.transform, x))
end

function kernelmatrix(κ::TransformedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(κ.kernel, _map(κ.transform, x), _map(κ.transform, y))
end
