"""
    ScaledKernel(k::Kernel, σ²::Real=1.0)

Scaled kernel derived from `k` by multiplication with variance `σ²`.

# Definition

For inputs ``x, x'``, the scaled kernel ``\\widetilde{k}`` derived from kernel ``k`` by
multiplication with variance ``\\sigma^2 > 0`` is defined as
```math
\\widetilde{k}(x, x'; k, \\sigma^2) = \\sigma^2 k(x, x').
```
"""
struct ScaledKernel{Tk<:Kernel,Tσ²<:Real} <: Kernel
    kernel::Tk
    σ²::Vector{Tσ²}
end

function ScaledKernel(kernel::Tk, σ²::Tσ²=1.0) where {Tk<:Kernel,Tσ²<:Real}
    @check_args(ScaledKernel, σ², σ² > zero(Tσ²), "σ² > 0")
    return ScaledKernel{Tk,Tσ²}(kernel, [σ²])
end

@functor ScaledKernel

(k::ScaledKernel)(x, y) = only(k.σ²) * k.kernel(x, y)

function kernelmatrix(κ::ScaledKernel, x::AbstractVector, y::AbstractVector)
    return only(κ.σ²) * kernelmatrix(κ.kernel, x, y)
end

function kernelmatrix(κ::ScaledKernel, x::AbstractVector)
    return only(κ.σ²) * kernelmatrix(κ.kernel, x)
end

function kernelmatrix_diag(κ::ScaledKernel, x::AbstractVector)
    return only(κ.σ²) * kernelmatrix_diag(κ.kernel, x)
end

function kernelmatrix_diag(κ::ScaledKernel, x::AbstractVector, y::AbstractVector)
    return only(κ.σ²) * kernelmatrix_diag(κ.kernel, x, y)
end

function kernelmatrix!(
    K::AbstractMatrix, κ::ScaledKernel, x::AbstractVector, y::AbstractVector
)
    kernelmatrix!(K, κ.kernel, x, y)
    K .*= κ.σ²
    return K
end

function kernelmatrix!(K::AbstractMatrix, κ::ScaledKernel, x::AbstractVector)
    kernelmatrix!(K, κ.kernel, x)
    K .*= κ.σ²
    return K
end

function kernelmatrix_diag!(K::AbstractVector, κ::ScaledKernel, x::AbstractVector)
    kernelmatrix_diag!(K, κ.kernel, x)
    K .*= κ.σ²
    return K
end

function kernelmatrix_diag!(
    K::AbstractVector, κ::ScaledKernel, x::AbstractVector, y::AbstractVector
)
    kernelmatrix_diag!(K, κ.kernel, x, y)
    K .*= κ.σ²
    return K
end

Base.:*(w::Real, k::Kernel) = ScaledKernel(k, w)

Base.show(io::IO, κ::ScaledKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::ScaledKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    return print(io, "\n" * ("\t"^(shift + 1)) * "- σ² = $(only(κ.σ²))")
end
