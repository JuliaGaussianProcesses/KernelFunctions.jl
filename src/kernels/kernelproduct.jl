"""
    KernelProduct(kernels)

Create a product of kernels.
One can also use the operator `*` :
```
    k1 = SqExponentialKernel()
    k2 = LinearKernel()
    k = KernelProduct([k1, k2]) == k1 * k2
    kernelmatrix(k, X) == kernelmatrix(k1, X) .* kernelmatrix(k2, X)
    kernelmatrix(k, X) == kernelmatrix(k1 * k2, X)
```
"""
struct KernelProduct{K} <: Kernel
    kernels::K
end

function KernelProduct(kernel::Kernel, kernels::Kernel...)
    return KernelProduct((kernel, kernels...))
end

Base.:*(k1::Kernel,k2::Kernel) = KernelProduct(k1, k2)
Base.:*(k1::KernelProduct,k2::KernelProduct) = KernelProduct(k1.kernels..., k2.kernels...) #TODO Add test
Base.:*(k::Kernel,kp::KernelProduct) = KernelProduct(k, kp.kernels...)
Base.:*(kp::KernelProduct,k::Kernel) = KernelProduct(kp.kernels..., k)

Base.length(k::KernelProduct) = length(k.kernels)

(κ::KernelProduct)(x, y) = prod(k(x, y) for k in κ.kernels)

function kernelmatrix(κ::KernelProduct, x::AbstractVector)
    return reduce(hadamard, kernelmatrix(κ.kernels[i], x) for i in 1:length(κ))
end

function kernelmatrix(κ::KernelProduct, x::AbstractVector, y::AbstractVector)
    return reduce(hadamard, kernelmatrix(κ.kernels[i], x, y) for i in 1:length(κ))
end

function kerneldiagmatrix(κ::KernelProduct, x::AbstractVector)
    return reduce(hadamard, kerneldiagmatrix(κ.kernels[i], x) for i in 1:length(κ))
end

function Base.show(io::IO, κ::KernelProduct)
    printshifted(io, κ, 0)
end

function printshifted(io::IO, κ::KernelProduct, shift::Int)
    print(io, "Product of $(length(κ)) kernels:")
    for i in 1:length(κ)
        print(io, "\n" * ("\t" ^ (shift + 1))* "- ")
        printshifted(io, κ.kernels[i], shift + 2)
    end
end
