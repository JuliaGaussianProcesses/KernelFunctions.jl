"""
    KernelSum(kernel, kernels..)

Create a sum of kernels. One can also use the operator `+`

```jldoctest
julia> k1 = SqExponentialKernel();

julia> k2 = LinearKernel();

julia> k = KernelSum(k1, k2) == k1 + k2
true

julia> kernelmatrix(k, X) == kernelmatrix(k1, X) .+ kernelmatrix(k2, X)
true

julia> kernelmatrix(k, X) == kernelmatrix(k1 + k2, X)
true
```
"""
struct KernelSum{Tk} <: Kernel
    kernels::Tk
end

function KernelSum(kernel::Kernel, kernels::Kernel...)
    return KernelSum((kernel, kernels...))
end

Base.:+(k1::Kernel, k2::Kernel) = KernelSum(k1, k2)
Base.:+(k1::KernelSum, k2::KernelSum) = KernelSum(k1.kernels..., k2.kernels...)
Base.:+(k::Kernel, ks::KernelSum) = KernelSum(k, ks.kernels...)

Base.length(k::KernelSum) = length(k.kernels)

(κ::KernelSum)(x, y) = sum(κ.kernels[i](x, y) for i in 1:length(κ))

function kernelmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kernelmatrix(κ.kernels[i], x) for i in 1:length(κ))
end

function kernelmatrix(κ::KernelSum, x::AbstractVector, y::AbstractVector)
    return sum(kernelmatrix(κ.kernels[i], x, y) for i in 1:length(κ))
end

function kerneldiagmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kerneldiagmatrix(κ.kernels[i], x) for i in 1:length(κ))
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
