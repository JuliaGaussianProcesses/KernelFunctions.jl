"""
    KernelSum(kernel, kernels..)

Create a sum of kernels. One can also use the operator `+`
```
    k1 = SqExponentialKernel()
    k2 = LinearKernel()
    k = KernelSum([k1, k2]) == k1 + k2
    kernelmatrix(k, X) == kernelmatrix(k1, X) .+ kernelmatrix(k2, X)
    kernelmatrix(k, X) == kernelmatrix(k1 + k2, X)
```
"""
struct KernelSum{Tk} <: Kernel
    kernels::Tk
end

function KernelSum(kernel::Kernel, kernels::Kernel...)
    return KernelSum((kernel, kernels...))
end

Base.:+(k1::Kernel, k2::Kernel) = KernelSum(k1, k2)
Base.:+(k1::ScaledKernel, k2::ScaledKernel) = KernelSum(k1, k2)
Base.:+(k1::KernelSum, k2::KernelSum) =
    KernelSum(k1.kernels..., k2.kernels...)
Base.:+(k::Kernel, ks::KernelSum) =
    KernelSum(k, ks.kernels...)
Base.:+(k::ScaledKernel, ks::KernelSum) =
        KernelSum(k, ks.kernels...)
Base.:+(k::ScaledKernel, ks::Kernel) =
        KernelSum(k, ks)
Base.:+(ks::KernelSum, k::Kernel) =
    KernelSum(ks.kernels..., k)
Base.:+(ks::KernelSum, k::ScaledKernel) =
        KernelSum(ks.kernels..., k)
Base.:+(ks::Kernel, k::ScaledKernel) =
        KernelSum(ks, k)

Base.length(k::KernelSum) = length(k.kernels)

(κ::KernelSum)(x, y) = sum(κ.kernels[i](x, y) for i in 1:length(κ))

function kernelmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kernelmatrix(κ.kernels[i], x) for i in 1:length(κ))
end

function kernelmatrix(κ::KernelSum, x::AbstractVector, y::AbstractVector)
    return sum(kernelmatrix(κ.kernels[i], x, y) for i in 1:length(κ))
end

function kerneldiagmatrix(κ::KernelSum, x::AbstractVector)
    return sum( kerneldiagmatrix(κ.kernels[i], x) for i in 1:length(κ))
end

function Base.show(io::IO, κ::KernelSum)
    printshifted(io, κ, 0)
end

function printshifted(io::IO,κ::KernelSum, shift::Int)
    print(io,"Sum of $(length(κ)) kernels:")
    for i in 1:length(κ)
        print(io, "\n" * ("\t" ^ (shift + 1)))
        printshifted(io, κ.kernels[i], shift + 2)
    end
end
