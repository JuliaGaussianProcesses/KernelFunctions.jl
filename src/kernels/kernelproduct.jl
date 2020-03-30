"""
    KernelProduct(k1::Kernel, k2::Kernel)

Create a product of kernels.
One can also use the operator `*` :
```
    k1 = SqExponentialKernel()
    k2 = LinearKernel()
    k = KernelProduct(k1, k2) == k1 * k2
    kernelmatrix(k, X) == kernelmatrix(k1, X) .* kernelmatrix(k2, X)
    kernelmatrix(k, X) == kernelmatrix(k1 * k2, X)
```
"""
struct KernelProduct{K₁<:Kernel, K₂<:Kernel} <: Kernel
    κ₁::K₁
    κ₂::K₂
end

Base.:*(k1::Kernel, k2::Kernel) = KernelProduct(k1, k2)

kappa(κ::KernelProduct, x ,y) = kappa(κ.κ₁, x, y) * kappa(κ.κ₁, x, y)

hadamard(x,y) = x.*y

function kernelmatrix(κ::KernelProduct, X::AbstractMatrix; obsdim::Int = defaultobs)
    kernelmatrix(κ.κ₁, X, obsdim = obsdim) .* kernelmatrix(κ.κ₂, X, obsdim = obsdim)
end

function kernelmatrix(
    κ::KernelProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int=defaultobs)
    kernelmatrix(κ.κ₁, X, Y, obsdim = obsdim) .* kernelmatrix(κ.κ₂, X, Y, obsdim = obsdim)
end

function kerneldiagmatrix(
    κ::KernelProduct,
    X::AbstractMatrix;
    obsdim::Int=defaultobs) #TODO Add test
    kerneldiagmatrix(κ.κ₁, X, obsdim = obsdim) .* kerneldiagmatrix(κ.κ₂, X, obsdim = obsdim)
end

function Base.show(io::IO, κ::KernelProduct)
    printshifted(io, κ, 0)
end

function printshifted(io::IO, κ::KernelProduct, shift::Int)
    print(io, "Kernel Product :")
    print(io, "\n" * ("\t" ^ (shift + 1)))
    printshifted(io, κ.κ₁, shift + 2)
    print(io, "\n" * ("\t" ^ (shift + 1)))
    printshifted(io, κ.κ₂, shift + 2)
end
