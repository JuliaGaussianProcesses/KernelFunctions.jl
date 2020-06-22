"""
    TransformedKernel(k::Kernel,t::Transform)

Return a kernel where inputs are pretransformed by `t` : `k(t(x),t(x'))`
Can also be called via [transform](@ref) : `transform(k, t)`
"""
struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

(k::TransformedKernel)(x, y) = k.kernel(k.transform(x), k.transform(y))

# Optimizations for scale transforms of simple kernels to save allocations:
# Instead of a multiplying every element of the inputs before evaluating the metric,
# we perform a scalar multiplcation of the distance of the original inputs, if possible.
function (k::TransformedKernel{<:SimpleKernel,<:ScaleTransform})(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
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
```julia
    transform(k::Kernel, t::Transform) (1)
    transform(k::Kernel, ρ::Real) (2)
    transform(k::Kernel, ρ::AbstractVector) (3)
```
(1) Create a TransformedKernel with transform `t` and kernel `k`
(2) Same as (1) with a `ScaleTransform` with scale `ρ`
(3) Same as (1) with an `ARDTransform` with scales `ρ`
"""
transform

transform(k::Kernel, t::Transform) = TransformedKernel(k, t)

function transform(k::TransformedKernel, t::Transform)
    return TransformedKernel(k.kernel, t ∘ k.transform)
end

transform(k::Kernel, ρ::Real) = transform(k, ScaleTransform(ρ))

transform(k::Kernel, ρ::AbstractVector) = transform(k, ARDTransform(ρ))

kernel(κ) = κ.kernel

Base.show(io::IO, κ::TransformedKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::TransformedKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    print(io,"\n" * ("\t" ^ (shift + 1)) * "- $(κ.transform)")
end

# Kernel matrix operations

function kerneldiagmatrix!(K::AbstractVector, κ::TransformedKernel, x::AbstractVector)
    return kerneldiagmatrix!(K, κ.kernel, map(κ.transform, x))
end

function kernelmatrix!(K::AbstractMatrix, κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix!(K, kernel(κ), map(κ.transform, x))
end

function kernelmatrix!(
    K::AbstractMatrix,
    κ::TransformedKernel,
    x::AbstractVector,
    y::AbstractVector,
)
    return kernelmatrix!(K, kernel(κ), map(κ.transform, x), map(κ.transform, y))
end

function kerneldiagmatrix(κ::TransformedKernel, x::AbstractVector)
    return kerneldiagmatrix(κ.kernel, map(κ.transform, x))
end

function kernelmatrix(κ::TransformedKernel, x::AbstractVector)
    return kernelmatrix(kernel(κ), map(κ.transform, x))
end

function kernelmatrix(κ::TransformedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(kernel(κ), map(κ.transform, x), map(κ.transform, y))
end
