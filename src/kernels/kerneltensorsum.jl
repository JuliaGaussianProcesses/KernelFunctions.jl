"""
    KernelTensorSum

Tensor sum of kernels.

# Definition

For inputs ``x = (x_1, \\ldots, x_n)`` and ``x' = (x'_1, \\ldots, x'_n)``, the tensor
sum of kernels ``k_1, \\ldots, k_n`` is defined as
```math
k(x, x'; k_1, \\ldots, k_n) = \\sum_{i=1}^n k_i(x_i, x'_i).
```

# Construction

The simplest way to specify a `KernelTensorSum` is to use the `⊕` operator (can be typed by `\\oplus<tab>`).
```jldoctest tensorproduct
julia> k1 = SqExponentialKernel(); k2 = LinearKernel(); X = rand(5, 2);

julia> kernelmatrix(k1 ⊕ k2, RowVecs(X)) == kernelmatrix(k1, X[:, 1]) + kernelmatrix(k2, X[:, 2])
true
```

You can also specify a `KernelTensorSum` by providing kernels as individual arguments
or as an iterable data structure such as a `Tuple` or a `Vector`. Using a tuple or
individual arguments guarantees that `KernelTensorSum` is concretely typed but might
lead to large compilation times if the number of kernels is large.
```jldoctest tensorproduct
julia> KernelTensorSum(k1, k2) == k1 ⊕ k2
true

julia> KernelTensorSum((k1, k2)) == k1 ⊕ k2
true

julia> KernelTensorSum([k1, k2]) == k1 ⊕ k2
true
```
"""
struct KernelTensorSum{K} <: Kernel
    kernels::K
end

function KernelTensorSum(kernel::Kernel, kernels::Kernel...)
    return KernelTensorSum((kernel, kernels...))
end

@functor KernelTensorSum

Base.length(kernel::KernelTensorSum) = length(kernel.kernels)

function (kernel::KernelTensorSum)(x, y)
    if !(length(x) == length(y) == length(kernel))
        throw(DimensionMismatch("number of kernels and number of features
are not consistent"))
    end
    return sum(k(xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

function validate_domain(k::KernelTensorSum, x::AbstractVector)
    return dim(x) == length(k) ||
           error("number of kernels and groups of features are not consistent")
end

function kernelmatrix(k::KernelTensorSum, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, +, k.kernels, slices(x))
end

function kernelmatrix(k::KernelTensorSum, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, +, k.kernels, slices(x), slices(y))
end

function kernelmatrix_diag(k::KernelTensorSum, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix_diag, +, k.kernels, slices(x))
end

function kernelmatrix_diag(k::KernelTensorSum, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix_diag, +, k.kernels, slices(x), slices(y))
end

function Base.:(==)(x::KernelTensorSum, y::KernelTensorSum)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

Base.show(io::IO, kernel::KernelTensorSum) = printshifted(io, kernel, 0)

function printshifted(io::IO, kernel::KernelTensorSum, shift::Int)
    print(io, "Tensor sum of ", length(kernel), " kernels:")
    for k in kernel.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
