"""
    NormalizedKernel(k::Kernel)

A normalized kernel derived from `k`.

# Definition

For inputs ``x, x'``, the normalized kernel ``\\widetilde{k}`` derived from
kernel ``k`` is defined as
```math
\\widetilde{k}(x, x'; k) = \\frac{k(x, x')}{\\sqrt{k(x, x) k(x', x')}}.
```
"""
struct NormalizedKernel{Tk<:Kernel} <: Kernel
    kernel::Tk
end

@functor NormalizedKernel

function ParameterHandling.flatten(::Type{T}, k::NormalizedKernel) where {T<:Real}
    vec, back = flatten(T, k.kernel)
    unflatten_to_normalizedkernel(v::Vector{T}) = NormalizedKernel(back(v))
    return vec, unflatten_to_normalizedkernel
end

(κ::NormalizedKernel)(x, y) = κ.kernel(x, y) / sqrt(κ.kernel(x, x) * κ.kernel(y, y))

function kernelmatrix(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    x_diag = kernelmatrix_diag(κ.kernel, x)
    x_diag_wide = x_diag * ones(eltype(x_diag), 1, length(y)) # ad perf hack. Is unit tested
    y_diag = kernelmatrix_diag(κ.kernel, y)
    y_diag_wide = y_diag * ones(eltype(y_diag), 1, length(x)) # ad perf hack. Is unit tested
    return kernelmatrix(κ.kernel, x, y) ./ sqrt.(x_diag_wide .* y_diag_wide')
end

function kernelmatrix(κ::NormalizedKernel, x::AbstractVector)
    x_diag = kernelmatrix_diag(κ.kernel, x)
    x_diag_wide = x_diag * ones(eltype(x_diag), 1, length(x)) # ad perf hack. Is unit tested
    return kernelmatrix(κ.kernel, x) ./ sqrt.(x_diag_wide .* x_diag_wide')
end

function kernelmatrix_diag(κ::NormalizedKernel, x::AbstractVector)
    first_x = first(x)
    return Ones{typeof(κ(first_x, first_x))}(length(x))
end

function kernelmatrix_diag(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix_diag(κ.kernel, x, y) ./
           sqrt.(kernelmatrix_diag(κ.kernel, x) .* kernelmatrix_diag(κ.kernel, y))
end

function kernelmatrix!(
    K::AbstractMatrix, κ::NormalizedKernel, x::AbstractVector, y::AbstractVector
)
    kernelmatrix!(K, κ.kernel, x, y)
    K ./=
        sqrt.(kernelmatrix_diag(κ.kernel, x) .* permutedims(kernelmatrix_diag(κ.kernel, y)))
    return K
end

function kernelmatrix!(K::AbstractMatrix, κ::NormalizedKernel, x::AbstractVector)
    kernelmatrix!(K, κ.kernel, x)
    x_diag = kernelmatrix_diag(κ.kernel, x)
    K ./= sqrt.(x_diag .* permutedims(x_diag))
    return K
end

function kernelmatrix_diag!(
    K::AbstractVector, κ::NormalizedKernel, x::AbstractVector, y::AbstractVector
)
    kernelmatrix_diag!(K, κ.kernel, x, y)
    K ./= sqrt.(kernelmatrix_diag(κ.kernel, x) .* kernelmatrix_diag(κ.kernel, y))
    return K
end

function kernelmatrix_diag!(K::AbstractVector, κ::NormalizedKernel, x::AbstractVector)
    first_x = first(x)
    return fill!(K, κ(first_x, first_x))
end

Base.show(io::IO, κ::NormalizedKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::NormalizedKernel, shift::Int)
    println(io, "Normalized Kernel:")
    for _ in 1:(shift + 1)
        print(io, "\t")
    end
    return printshifted(io, κ.kernel, shift + 1)
end
