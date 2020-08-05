"""
    KernelSum <: Kernel

Create a sum of kernels. One can also use the operator `+`.

There are various ways in which you create a `KernelSum`:

The simplest way to specify a `KernelSum` would be to use the overloaded `+` operator. This is 
equivalent to creating a `KernelSum` by specifying the kernels as the arguments to the constructor.  
```jldoctest kernelsum
julia> k1 = SqExponentialKernel(); k2 = LinearKernel(); X = rand(5);

julia> (k = k1 + k2) == KernelSum(k1, k2)
true

julia> kernelmatrix(k1 + k2, X) == kernelmatrix(k1, X) .+ kernelmatrix(k2, X)
true

julia> kernelmatrix(k, X) == kernelmatrix(k1 + k2, X)
true
```

You could also specify a `KernelSum` by providing a `Tuple` or a `Vector` of the 
kernels to be summed. We suggest you to use a `Tuple` when you have fewer components  
and a `Vector` when dealing with a large number of components.
```jldoctest kernelsum
julia> KernelSum((k1, k2)) == k1 + k2
true

julia> KernelSum([k1, k2]) == KernelSum((k1, k2)) == k1 + k2
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

function Base.:+(
    k1::KernelSum{<:AbstractVector{<:Kernel}}, 
    k2::KernelSum{<:AbstractVector{<:Kernel}}
)
    KernelSum(vcat(k1.kernels, k2.kernels))
end

Base.:+(k1::KernelSum, k2::KernelSum) = KernelSum(k1.kernels..., k2.kernels...)

function Base.:+(k::Kernel, ks::KernelSum{<:AbstractVector{<:Kernel}})
    return KernelSum(vcat(k, ks.kernels))
end

Base.:+(k::Kernel, ks::KernelSum) = KernelSum(k, ks.kernels...)

function Base.:+(ks::KernelSum{<:AbstractVector{<:Kernel}}, k::Kernel)
    return KernelSum(vcat(ks.kernels, k))
end

Base.:+(ks::KernelSum, k::Kernel) = KernelSum(ks.kernels..., k)

Base.length(k::KernelSum) = length(k.kernels)

(κ::KernelSum)(x, y) = sum(k(x, y) for k in κ.kernels)

function kernelmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kernelmatrix(k, x) for k in κ.kernels)
end

function kernelmatrix(κ::KernelSum, x::AbstractVector, y::AbstractVector)
    return sum(kernelmatrix(k, x, y) for k in κ.kernels)
end

function kerneldiagmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kerneldiagmatrix(k, x) for k in κ.kernels)
end

function Base.show(io::IO, κ::KernelSum)
    printshifted(io, κ, 0)
end

function Base.:(==)(x::KernelSum, y::KernelSum)
    return (
        length(x.kernels) == length(y.kernels) && 
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

function printshifted(io::IO,κ::KernelSum, shift::Int)
    print(io,"Sum of $(length(κ)) kernels:")
    for k in κ.kernels
        print(io, "\n" )
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
