"""
    TensorProduct(kernels...)

Create a tensor product of kernels.
"""
struct TensorProduct{K} <: Kernel
    kernels::K
end

function TensorProduct(kernel::Kernel, kernels::Kernel...)
    return TensorProduct((kernel, kernels...))
end

Base.length(kernel::TensorProduct) = length(kernel.kernels)

(kernel::TensorProduct)(x, y) = kappa(kernel, x, y)
function kappa(kernel::TensorProduct, x, y)
    return prod(kappa(k, xi, yi) for (k, xi, yi) in zip(kernel.kernels, x, y))
end

# TODO: General implementation of `kernelmatrix` and `kerneldiagmatrix`
# Default implementation assumes 1D observations

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1, 2) || "obsdim should be 1 or 2 (see docs of kernelmatrix))"

    featuredim = feature_dim(obsdim)
    if !check_dims(K, X, X, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernelmatrix!(K, kernel.kernels[1], selectdim(X, featuredim, 1))
    for (k, Xi) in Iterators.drop(zip(kernel.kernels, eachslice(X; dims = featuredim)), 1)
        K .*= kernelmatrix(k, Xi)
    end

    return K
end

function kernelmatrix!(
    K::AbstractMatrix,
    kernel::TensorProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(K, X, Y, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not consistent with X ($(size(X))) and Y ($(size(Y)))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kernelmatrix!(K, kernel.kernels[1], selectdim(X, featuredim, 1),
                  selectdim(Y, featuredim, 1))
    for (k, Xi, Yi) in Iterators.drop(zip(kernel.kernels,
                                          eachslice(X; dims = featuredim),
                                          eachslice(Y; dims = featuredim)), 1)
        K .*= kernelmatrix(k, Xi, Yi)
    end

    return K
end

# mapreduce with multiple iterators requires Julia 1.2 or later.

function kernelmatrix(
    kernel::TensorProduct,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(X, X, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
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
    @assert obsdim ∈ (1, 2) || error("obsdim should be 1 or 2 (see docs of kernelmatrix))")

    featuredim = feature_dim(obsdim)
    if !check_dims(X, Y, featuredim, obsdim)
        throw(DimensionMismatch("Dimensions $(size(K)) of the target array K are not consistent with X ($(size(X))) and Y ($(size(Y)))"))
    end

    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    return mapreduce((x, y) -> x .* y,
                     zip(kernel.kernels,
                         eachslice(X; dims = featuredim),
                         eachslice(Y; dims = featuredim))) do (k, Xi, Yi)
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
        throw(DimensionMismatch("Dimensions of the target array K $(size(K)) are not consistent with X $(size(X))"))
    end

    featuredim = feature_dim(obsdim)
    size(X, featuredim) == length(kernel) ||
        error("number of kernels and groups of features are not consistent")

    kerneldiagmatrix!(K, kernel.kernels[1], selectdim(X, featuredim, 1))
    for (k, Xi) in Iterators.drop(zip(kernel.kernels, eachslice(X; dims = featuredim)), 1)
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

    return mapreduce((x, y) -> x .* y,
                     zip(kernel.kernels, eachslice(X; dims = featuredim))) do (k, Xi)
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
