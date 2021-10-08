"""
    KernelTensorProduct

Tensor product of kernels.

# Definition

For inputs ``x = (x_1, \\ldots, x_n)`` and ``x' = (x'_1, \\ldots, x'_n)``, the tensor
product of kernels ``k_1, \\ldots, k_n`` is defined as
```math
k(x, x'; k_1, \\ldots, k_n) = \\Big(\\bigotimes_{i=1}^n k_i\\Big)(x, x') = \\prod_{i=1}^n k_i(x_i, x'_i).
```

# Construction

The simplest way to specify a `KernelTensorProduct` is to use the overloaded `tensor`
operator or its alias `⊗` (can be typed by `\\otimes<tab>`).
```jldoctest tensorproduct
julia> k1 = SqExponentialKernel(); k2 = LinearKernel(); X = rand(5, 2);

julia> kernelmatrix(k1 ⊗ k2, RowVecs(X)) == kernelmatrix(k1, X[:, 1]) .* kernelmatrix(k2, X[:, 2])
true
```

You can also specify a `KernelTensorProduct` by providing kernels as individual arguments
or as an iterable data structure such as a `Tuple` or a `Vector`. Using a tuple or
individual arguments guarantees that `KernelTensorProduct` is concretely typed but might
lead to large compilation times if the number of kernels is large.
```jldoctest tensorproduct
julia> KernelTensorProduct(k1, k2) == k1 ⊗ k2
true

julia> KernelTensorProduct((k1, k2)) == k1 ⊗ k2
true

julia> KernelTensorProduct([k1, k2]) == k1 ⊗ k2
true
```
"""
struct KernelTensorProduct{K} <: Kernel
    kernels::K
end

function KernelTensorProduct(kernel::Kernel, kernels::Kernel...)
    return KernelTensorProduct((kernel, kernels...))
end

@functor KernelTensorProduct

Base.length(kernel::KernelTensorProduct) = length(kernel.kernels)

function (kernel::KernelTensorProduct)(x, y)
    if !(length(x) == length(y) == length(kernel))
        throw(DimensionMismatch("number of kernels and number of features
are not consistent"))
    end
    return prod(k(xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

function validate_domain(k::KernelTensorProduct, x::AbstractVector)
    return dim(x) == length(k) ||
           error("number of kernels and groups of features are not consistent")
end

# Utility for slicing up inputs.
slices(x::AbstractVector{<:Real}) = (x,)
slices(x::ColVecs) = eachrow(x.X)
slices(x::RowVecs) = eachcol(x.X)

function kernelmatrix!(K::AbstractMatrix, k::KernelTensorProduct, x::AbstractVector)
    validate_inplace_dims(K, x)
    validate_domain(k, x)

    first_kernels, tail_kernels = Iterators.peel(kernels)
    first_x, tail_x = Iterators.peel(slices(x))

    # handle first kernel and input
    kernelmatrix!(K, first_kernels, first_x)

    # handle remaining kernels and inputs
    Ktmp = similar(K)
    for (k, xi) in zip(tail_kernels, tail_x)
        kernelmatrix!(Ktmp, k, xi)
        hadamard!(K, K, Ktmp)
    end

    return K
end

function kernelmatrix!(
    K::AbstractMatrix, k::KernelTensorProduct, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    validate_domain(k, x)

    first_kernels, tail_kernels = Iterators.peel(kernels)
    first_x, tail_x = Iterators.peel(slices(x))
    first_y, tail_y = Iterators.peel(slices(y))

    # handle first kernel and inputs
    kernelmatrix!(K, first_kernels, first_x, first_y)

    # handle remaining kernels and inputs
    Ktmp = similar(K)
    for (k, xi, yi) in zip(tail_kernels, tail_x, tail_y)
        kernelmatrix!(Ktmp, k, xi, yi)
        hadamard!(K, K, Ktmp)
    end

    return K
end

function kernelmatrix_diag!(K::AbstractVector, k::KernelTensorProduct, x::AbstractVector)
    validate_inplace_dims(K, x)
    validate_domain(k, x)

    first_kernels, tail_kernels = Iterators.peel(kernels)
    first_x, tail_x = Iterators.peel(slices(x))

    # handle first kernel and input
    kernelmatrix_diag!(K, first_kernels, first_x)

    # handle remaining kernels and inputs
    Ktmp = similar(K)
    for (k, xi) in zip(tail_kernels, tail_x)
        kernelmatrix_diag!(Ktmp, k, xi)
        hadamard!(K, K, Ktmp)
    end

    return K
end

function kernelmatrix_diag!(
    K::AbstractVector, k::KernelTensorProduct, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    validate_domain(k, x)

    first_kernels, tail_kernels = Iterators.peel(kernels)
    first_x, tail_x = Iterators.peel(slices(x))
    first_y, tail_y = Iterators.peel(slices(y))

    # handle first kernel and inputs
    kernelmatrix_diag!(K, first_kernels, first_x, first_y)

    # handle remaining kernels and inputs
    Ktmp = similar(K)
    for (k, xi, yi) in zip(tail_kernels, tail_x, tail_y)
        kernelmatrix_diag!(Ktmp, k, xi, yi)
        hadamard!(K, K, Ktmp)
    end

    return K
end

function kernelmatrix(k::KernelTensorProduct, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, hadamard, k.kernels, slices(x))
end

function kernelmatrix(k::KernelTensorProduct, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, hadamard, k.kernels, slices(x), slices(y))
end

function kernelmatrix_diag(k::KernelTensorProduct, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix_diag, hadamard, k.kernels, slices(x))
end

function kernelmatrix_diag(k::KernelTensorProduct, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix_diag, hadamard, k.kernels, slices(x), slices(y))
end

Base.show(io::IO, kernel::KernelTensorProduct) = printshifted(io, kernel, 0)

function Base.:(==)(x::KernelTensorProduct, y::KernelTensorProduct)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

function printshifted(io::IO, kernel::KernelTensorProduct, shift::Int)
    print(io, "Tensor product of ", length(kernel), " kernels:")
    for k in kernel.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
