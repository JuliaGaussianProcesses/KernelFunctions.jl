"""
    TensorProduct(kernels...)

Create a tensor product kernel from kernels ``k_1, \\ldots, k_n``, i.e.,
a kernel ``k`` that is given by
```math
k(x, y) = \\prod_{i=1}^n k_i(x_i, y_i).
```

The `kernels` can be specified as individual arguments, a tuple, or an iterable data
structure such as an array. Using a tuple or individual arguments guarantees that
`TensorProduct` is concretely typed but might lead to large compilation times if the
number of kernels is large.
"""
struct TensorProduct{K} <: Kernel
    kernels::K
end

function TensorProduct(kernel::Kernel, kernels::Kernel...)
    return TensorProduct((kernel, kernels...))
end

@functor TensorProduct

Base.length(kernel::TensorProduct) = length(kernel.kernels)

function (kernel::TensorProduct)(x, y)
    if !(length(x) == length(y) == length(kernel))
        throw(DimensionMismatch("number of kernels and number of features
are not consistent"))
    end
    return prod(k(xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

function validate_domain(k::TensorProduct, x::AbstractVector)
    return dim(x) == length(k) ||
        error("number of kernels and groups of features are not consistent")
end

# Utility for slicing up inputs.
slices(x::AbstractVector{<:Real}) = (x,)
slices(x::ColVecs) = eachrow(x.X)
slices(x::RowVecs) = eachcol(x.X)

function kernelmatrix!(K::AbstractMatrix, k::TensorProduct, x::AbstractVector)
    validate_inplace_dims(K, x)
    validate_domain(k, x)

    kernels_and_inputs = zip(k.kernels, slices(x))
    kernelmatrix!(K, first(kernels_and_inputs)...)
    for (k, xi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kernelmatrix(k, xi)
    end

    return K
end

function kernelmatrix!(
    K::AbstractMatrix, k::TensorProduct, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    validate_domain(k, x)

    kernels_and_inputs = zip(k.kernels, slices(x), slices(y))
    kernelmatrix!(K, first(kernels_and_inputs)...)
    for (k, xi, yi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kernelmatrix(k, xi, yi)
    end

    return K
end

function kerneldiagmatrix!(K::AbstractVector, k::TensorProduct, x::AbstractVector)
    validate_inplace_dims(K, x)
    validate_domain(k, x)

    kernels_and_inputs = zip(k.kernels, slices(x))
    kerneldiagmatrix!(K, first(kernels_and_inputs)...)
    for (k, xi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kerneldiagmatrix(k, xi)
    end

    return K
end

function kernelmatrix(k::TensorProduct, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, hadamard, k.kernels, slices(x))
end

function kernelmatrix(k::TensorProduct, x::AbstractVector, y::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kernelmatrix, hadamard, k.kernels, slices(x), slices(y))
end

function kerneldiagmatrix(k::TensorProduct, x::AbstractVector)
    validate_domain(k, x)
    return mapreduce(kerneldiagmatrix, hadamard, k.kernels, slices(x))
end

Base.show(io::IO, kernel::TensorProduct) = printshifted(io, kernel, 0)

function Base.:(==)(x::TensorProduct, y::TensorProduct)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

function printshifted(io::IO, kernel::TensorProduct, shift::Int)
    print(io, "Tensor product of ", length(kernel), " kernels:")
    for k in kernel.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        print(io, "- ")
        printshifted(io, k, shift + 2)
    end
end
