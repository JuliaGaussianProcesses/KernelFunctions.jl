"""
    KernelProduct <: Kernel

Create a product of kernels. One can also use the overloaded operator `*`.

There are various ways in which you create a `KernelProduct`:

The simplest way to specify a `KernelProduct` would be to use the overloaded `*` operator. This is 
equivalent to creating a `KernelProduct` by specifying the kernels as the arguments to the constructor.  
```jldoctest kernelprod
julia> k1 = SqExponentialKernel(); k2 = LinearKernel(); X = rand(5);

julia> (k = k1 * k2) == KernelProduct(k1, k2)
true

julia> kernelmatrix(k1 * k2, X) == kernelmatrix(k1, X) .* kernelmatrix(k2, X)
true

julia> kernelmatrix(k, X) == kernelmatrix(k1 * k2, X)
true
```

You could also specify a `KernelProduct` by providing a `Tuple` or a `Vector` of the 
kernels to be multiplied. We suggest you to use a `Tuple` when you have fewer components  
and a `Vector` when dealing with a large number of components.
```jldoctest kernelprod
julia> KernelProduct((k1, k2)) == k1 * k2
true

julia> KernelProduct([k1, k2]) == KernelProduct((k1, k2)) == k1 * k2
true
```
"""
struct KernelProduct{Tk} <: Kernel
    kernels::Tk
end

function KernelProduct(kernel::Kernel, kernels::Kernel...)
    return KernelProduct((kernel, kernels...))
end

@functor KernelProduct

Base.:*(k1::Kernel, k2::Kernel) = KernelProduct(k1, k2)

function Base.:*(
    k1::KernelProduct{<:AbstractVector{<:Kernel}},
    k2::KernelProduct{<:AbstractVector{<:Kernel}},
)
    return KernelProduct(vcat(k1.kernels, k2.kernels))
end

function Base.:*(k1::KernelProduct, k2::KernelProduct)
    return KernelProduct(k1.kernels..., k2.kernels...)
end

function Base.:*(k::Kernel, ks::KernelProduct{<:AbstractVector{<:Kernel}})
    return KernelProduct(vcat(k, ks.kernels))
end

Base.:*(k::Kernel, kp::KernelProduct) = KernelProduct(k, kp.kernels...)

function Base.:*(ks::KernelProduct{<:AbstractVector{<:Kernel}}, k::Kernel)
    return KernelProduct(vcat(ks.kernels, k))
end

Base.:*(kp::KernelProduct, k::Kernel) = KernelProduct(kp.kernels..., k)

Base.length(k::KernelProduct) = length(k.kernels)

(κ::KernelProduct)(x, y) = prod(k(x, y) for k in κ.kernels)

function kernelmatrix(κ::KernelProduct, x::AbstractVector)
    return reduce(hadamard, kernelmatrix(k, x) for k in κ.kernels)
end

function kernelmatrix(κ::KernelProduct, x::AbstractVector, y::AbstractVector)
    return reduce(hadamard, kernelmatrix(k, x, y) for k in κ.kernels)
end

function kerneldiagmatrix(κ::KernelProduct, x::AbstractVector)
    return reduce(hadamard, kerneldiagmatrix(k, x) for k in κ.kernels)
end

function Base.show(io::IO, κ::KernelProduct)
    return printshifted(io, κ, 0)
end

function Base.:(==)(x::KernelProduct, y::KernelProduct)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

function printshifted(io::IO, κ::KernelProduct, shift::Int)
    print(io, "Product of $(length(κ)) kernels:")
    for k in κ.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
