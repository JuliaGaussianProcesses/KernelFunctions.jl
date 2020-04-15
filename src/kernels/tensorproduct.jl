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
