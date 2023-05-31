"""
    KernelIndependentSum

Independent sum of kernels.

# Definition

For inputs ``x = (x_1, \\ldots, x_n)`` and ``x' = (x'_1, \\ldots, x'_n)``, the tensor
sum of kernels ``k_1, \\ldots, k_n`` is defined as
```math
k(x, x'; k_1, \\ldots, k_n) = \\sum_{i=1}^n k_i(x_i, x'_i).
```

# Construction

The simplest way to specify a `KernelIndependentSum` is to use the `⊕` operator (can be typed by `\\oplus<tab>`).
```jldoctest independentsum
julia> k1 = SqExponentialKernel(); k2 = LinearKernel(); X = rand(5, 2);

julia> kernelmatrix(k1 ⊕ k2, RowVecs(X)) == kernelmatrix(k1, X[:, 1]) + kernelmatrix(k2, X[:, 2])
true
```

You can also specify a `KernelIndependentSum` by providing kernels as individual arguments
or as an iterable data structure such as a `Tuple` or a `Vector`. Using a tuple or
individual arguments guarantees that `KernelIndependentSum` is concretely typed but might
lead to large compilation times if the number of kernels is large.
```jldoctest independentsum
julia> KernelIndependentSum(k1, k2) == k1 ⊕ k2
true

julia> KernelIndependentSum((k1, k2)) == k1 ⊕ k2
true

julia> KernelIndependentSum([k1, k2]) == k1 ⊕ k2
true
```
"""
struct KernelIndependentSum{K} <: Kernel
    kernels::K
end

function KernelIndependentSum(kernel::Kernel, kernels::Kernel...)
    return KernelIndependentSum((kernel, kernels...))
end

@functor KernelIndependentSum

Base.length(kernel::KernelIndependentSum) = length(kernel.kernels)

function (kernel::KernelIndependentSum)(x, y)
    if !((nx = length(x)) == (ny = length(y)) == (nkernels = length(kernel)))
        throw(
            DimensionMismatch(
                "number of kernels ($nkernels) and number of features (x=$nx, y=$ny) are not consistent",
            ),
        )
    end
    return sum(k(xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

function validate_domain(k::KernelIndependentSum, x::AbstractVector, y::AbstractVector)
    return (dx = dim(x)) == (dy = dim(y)) == (nkernels = length(k)) || error(
        "number of kernels ($nkernels) and group of features (x=$dx), y=$dy) are not consistent",
    )
end

function validate_domain(k::KernelIndependentSum, x::AbstractVector)
    return validate_domain(k, x, x)
end

function kernelmatrix(k::KernelIndependentSum, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, +, k.kernels, slices(x))
end

function kernelmatrix(k::KernelIndependentSum, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x, y)
    return mapreduce(kernelmatrix, +, k.kernels, slices(x), slices(y))
end

function kernelmatrix_diag(k::KernelIndependentSum, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix_diag, +, k.kernels, slices(x))
end

function kernelmatrix_diag(k::KernelIndependentSum, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x, y)
    return mapreduce(kernelmatrix_diag, +, k.kernels, slices(x), slices(y))
end

function Base.:(==)(x::KernelIndependentSum, y::KernelIndependentSum)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

Base.show(io::IO, kernel::KernelIndependentSum) = printshifted(io, kernel, 0)

function printshifted(io::IO, kernel::KernelIndependentSum, shift::Int)
    print(io, "Independent sum of ", length(kernel), " kernels:")
    for k in kernel.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
