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
struct NormalizedKernel{Tk <: Kernel} <: Kernel
    kernel::Tk
end

@functor NormalizedKernel

(κ::NormalizedKernel)(x, y) = κ.kernel(x, y) / sqrt(κ.kernel(x, x) * κ.kernel(y, y))

function kernelmatrix(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix(κ.kernel, x, y) ./ _normalizationmatrix(κ, x, y)
end

function kernelmatrix(κ::NormalizedKernel, x::AbstractVector)
    return kernelmatrix(κ.kernel, x) ./ _normalizationmatrix(κ, x, x)
end

function kernelmatrix_diag(κ::NormalizedKernel, x::AbstractVector)
    return kernelmatrix_diag(κ.kernel, x) ./ _normalizationmatrix_diag(κ, x, x)
end

function kernelmatrix_diag(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    return kernelmatrix_diag(κ.kernel, x, y) ./ _normalizationmatrix_diag(κ, x, y)
end

function kernelmatrix!(
    K::AbstractMatrix, κ::NormalizedKernel, x::AbstractVector, y::AbstractVector
)
    kernelmatrix!(K, κ.kernel, x, y)
    K ./= _normalizationmatrix(κ, x, y)
    return K
end

function kernelmatrix!(K::AbstractMatrix, κ::NormalizedKernel, x::AbstractVector)
    kernelmatrix!(K, κ.kernel, x)
    K ./= _normalizationmatrix(κ, x, x)
    return K
end

function kernelmatrix_diag!(K::AbstractVector, κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    kernelmatrix_diag!(K, κ.kernel, x, y)
    K ./= _normalizationmatrix_diag(κ, x, y)
    return K
end

function kernelmatrix_diag!(K::AbstractVector, κ::NormalizedKernel, x::AbstractVector)
    kernelmatrix_diag!(K, κ.kernel, x)
    K ./= _normalizationmatrix_diag(κ, x, x)
    return K
end

# Calculate the required normalization factor for each element
function _normalizationmatrix(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    return sqrt.(kernelmatrix_diag(κ.kernel, x) .* permutedims(kernelmatrix_diag(κ.kernel, y)))
end

function _normalizationmatrix_diag(κ::NormalizedKernel, x::AbstractVector, y::AbstractVector)
    return sqrt.(kernelmatrix_diag(κ.kernel, x) .* kernelmatrix_diag(κ.kernel, y))
end

Base.show(io::IO, κ::NormalizedKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::NormalizedKernel, shift::Int)
    print(io, "Normalized Kernel:" * "\n")
    for _ in 1:(shift + 1)
        print(io, "\t")
    end
    return printshifted(io, κ.kernel, shift + 1)
end
