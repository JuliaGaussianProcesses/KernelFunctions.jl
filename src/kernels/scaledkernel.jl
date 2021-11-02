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
    σ²::Tσ²

    function ScaledKernel(kernel::Kernel, σ²::Real)
        @check_args(ScaledKernel, σ², σ² > zero(σ²), "σ² > 0")
        return new{typeof(kernel),typeof(σ²)}(kernel, σ²)
    end
end

ScaledKernel(kernel::Kernel) = ScaledKernel(kernel, 1.0)

# σ² is a positive parameter (and a scalar!) but Functors does not handle
# parameter constraints
@functor ScaledKernel (kernel,)

function ParameterHandling.flatten(::Type{T}, k::ScaledKernel{<:Kernel,S}) where {T<:Real,S<:Real}
    kernel_vec, kernel_back = flatten(T, k.kernel)
    function unflatten_to_scaledkernel(v::Vector{T})
        kernel = kernel_back(v[1:end-1])
        return ScaledKernel(kernel, S(exp(last(v))))
    end
    return vcat(kernel_vec, T(log(k.σ²))), unflatten_to_scaledkernel
end

(k::ScaledKernel)(x, y) = k.σ² * k.kernel(x, y)

function kernelmatrix(κ::ScaledKernel, x::AbstractVector, y::AbstractVector)
    return κ.σ² .* kernelmatrix(κ.kernel, x, y)
end

function kernelmatrix(κ::ScaledKernel, x::AbstractVector)
    return κ.σ² .* kernelmatrix(κ.kernel, x)
end

function kernelmatrix_diag(κ::ScaledKernel, x::AbstractVector)
    return κ.σ² .* kernelmatrix_diag(κ.kernel, x)
end

function kernelmatrix_diag(κ::ScaledKernel, x::AbstractVector, y::AbstractVector)
    return κ.σ² .* kernelmatrix_diag(κ.kernel, x, y)
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
    print(io, "\n")
    for _ in 1:(shift + 1)
        print(io, "\t")
    end
    print(io, "- σ² = ", κ.σ²)
end
