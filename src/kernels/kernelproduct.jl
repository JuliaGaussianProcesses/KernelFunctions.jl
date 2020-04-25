"""
    KernelProduct(kernels::Array{Kernel})

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
struct KernelProduct <: Kernel
    kernels::Vector{Kernel}
end

Base.:*(k1::Kernel,k2::Kernel) = KernelProduct([k1,k2])
Base.:*(k1::KernelProduct,k2::KernelProduct) = KernelProduct(vcat(k1.kernels,k2.kernels)) #TODO Add test
Base.:*(k::Kernel,kp::KernelProduct) = KernelProduct(vcat(k,kp.kernels))
Base.:*(kp::KernelProduct,k::Kernel) = KernelProduct(vcat(kp.kernels,k))

Base.length(k::KernelProduct) = length(k.kernels)

(κ::KernelProduct)(x, y) = prod(k(x, y) for k in κ.kernels)

hadamard(x,y) = x.*y

function kernelmatrix(
    κ::KernelProduct,
    X::AbstractMatrix;
    obsdim::Int=defaultobs,
)
    reduce(hadamard, kernelmatrix(κ.kernels[i], X, obsdim = obsdim) for i in 1:length(κ))
end

function kernelmatrix(
    κ::KernelProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    obsdim::Int=defaultobs,
)
    reduce(hadamard, kernelmatrix(κ.kernels[i], X, Y, obsdim = obsdim) for i in 1:length(κ))
end

function kerneldiagmatrix(
    κ::KernelProduct,
    X::AbstractMatrix;
    obsdim::Int=defaultobs,
) #TODO Add test
    reduce(hadamard, kerneldiagmatrix(κ.kernels[i], X, obsdim = obsdim) for i in 1:length(κ))
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
