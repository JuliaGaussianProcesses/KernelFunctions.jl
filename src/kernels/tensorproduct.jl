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

Base.length(kernel::TensorProduct) = length(kernel.kernels)

function kappa(kernel::TensorProduct, x, y)
    return prod(kappa(k, xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

# TODO: General implementation of `kernelmatrix` and `kerneldiagmatrix`
# Default implementation assumes 1D observations

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    obsdim ∈ (1, 2) || "obsdim should be 1 or 2 (see docs of kernelmatrix))"

    featuredim = feature_dim(obsdim)
    if !check_dims(K, X, X, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not " *
                                "consistent with X $(size(X))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernels_and_inputs = zip(kernel.kernels, eachslice(X; dims = featuredim))
    kernelmatrix!(K, first(kernels_and_inputs)...)
    for (k, Xi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kernelmatrix(k, Xi)
    end

    return K
end

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::TensorProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(K, X, Y, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not " *
                                "consistent with X ($(size(X))) and Y ($(size(Y)))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernels_and_inputs = zip(
        kernel.kernels,
        eachslice(X; dims = featuredim),
        eachslice(Y; dims = featuredim),
    )
    kernelmatrix!(K, first(kernels_and_inputs)...)
    for (k, Xi, Yi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kernelmatrix(k, Xi, Yi)
    end

    return K
end

# mapreduce with multiple iterators requires Julia 1.2 or later.

function kernelmatrix(
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(X, X, featuredim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not " *
                                "consistent with X $(size(X))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    return mapreduce((x, y) -> x .* y,
                     zip(kernel.kernels, eachslice(X; dims = featuredim))) do (k, Xi)
        kernelmatrix(k, Xi)
    end
end

function kernelmatrix(
    kernel::TensorProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(X, Y, featuredim)
        throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not " *
                                "consistent with X ($(size(X))) and Y ($(size(Y)))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernels_and_inputs = zip(
        kernel.kernels,
        eachslice(X; dims = featuredim),
        eachslice(Y; dims = featuredim),
    )
    return mapreduce((x, y) -> x .* y, kernels_and_inputs) do (k, Xi, Yi)
        kernelmatrix(k, Xi, Yi)
    end
end

function kerneldiagmatrix!(
    K::AbstractVector,
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")
    if length(K) != size(X, obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not " *
                                "consistent with X $(size(X))"))
    end

    featuredim = feature_dim(obsdim)
    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernels_and_inputs = zip(kernel.kernels, eachslice(X; dims = featuredim))
    kerneldiagmatrix!(K, first(kernels_and_inputs)...)
    for (k, Xi) in Iterators.drop(kernels_and_inputs, 1)
        K .*= kerneldiagmatrix(k, Xi)
    end

    return K
end

function kerneldiagmatrix(
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1,2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernels_and_inputs = zip(kernel.kernels, eachslice(X; dims = featuredim))
    return mapreduce((x, y) -> x .* y, kernels_and_inputs) do (k, Xi)
        kerneldiagmatrix(k, Xi)
    end
end

Base.show(io::IO, kernel::TensorProduct) = printshifted(io, kernel, 0)

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
