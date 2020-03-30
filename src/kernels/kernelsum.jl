"""
    KernelSum(k1::Kernel, k2::Kernel)

Create a positive weighted sum of kernels. All weights should be positive.
One can also use the operator `+`
```
    k1 = SqExponentialKernel()
    k2 = LinearKernel()
    k = KernelSum(k1, k2) == k1 + k2
    kernelmatrix(k, X) == kernelmatrix(k1, X) .+ kernelmatrix(k2, X)
    kernelmatrix(k, X) == kernelmatrix(k1 + k2, X)
```
"""
struct KernelSum{K₁<:Kernel, K₂<:Kernel} <: Kernel
    κ₁::K₁
    κ₂::K₂
end

Base.:+(k1::Kernel, k2::Kernel) = KernelSum(k1, k2)

nmetrics(κ::KernelSum) = metric(κ.κ₁) == metric(κ.κ₂) ? 2 : 1

kappa(κ::KernelSum, x, y) = kappa(κ.κ₁, x, y) + kappa(κ.κ₂, x, y)

function kernelmatrix(κ::KernelSum, X::AbstractMatrix; obsdim::Int = defaultobs)
    kernelmatrix(κ.κ₁, X, obsdim = obsdim) + kernelmatrix(κ.κ₂, X, obsdim = obsdim)
end

function kernelmatrix(
    κ::KernelSum,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    kernelmatrix(κ.κ₁, X, Y, obsdim = obsdim) + kernelmatrix(κ.κ₂, X, Y, obsdim = obsdim)
end

function kerneldiagmatrix(
    κ::KernelSum,
    X::AbstractMatrix;
    obsdim::Int = defaultobs,
)
    kerneldiagmatrix(κ.κ₁, X, obsdim = obsdim) + kerneldiagmatrix(κ.κ₂, X, obsdim = obsdim)
end

function Base.show(io::IO, κ::KernelSum)
    printshifted(io, κ, 0)
end

function printshifted(io::IO, κ::KernelSum, shift::Int)
    print(io,"Kernel Sum :")
    print(io, "\n" * ("\t" ^ (shift + 1)))
    printshifted(io, κ.κ₁, shift + 2)
    print(io, "\n" * ("\t" ^ (shift + 1)))
    printshifted(io, κ.κ₂, shift + 2)
end
