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

@functor KernelSum

function ParameterHandling.flatten(::Type{T}, k::KernelSum) where {T<:Real}
    vecs_and_backs = map(Base.Fix1(flatten, T), k.kernels)
    vecs = map(first, vecs_and_backs)
    length_vecs = map(length, vecs)
    backs = map(last, vecs_and_backs)
    flat_vecs = reduce(vcat, vecs)
    n = length(flat_vecs)
    function unflatten_to_kernelsum(v::Vector{T})
        length(v) == n || error("incorrect number of parameters")
        offset = Ref(0)
        kernels = map(backs, length_vecs) do back, length_vec
            oldoffset = offset[]
            newoffset = offset[] = oldoffset + length_vec
            return back(v[(oldoffset + 1):newoffset])
        end
        return KernelSum(kernels)
    end
    return flat_vecs, unflatten_to_kernelsum
end

Base.length(k::KernelSum) = length(k.kernels)

(κ::KernelSum)(x, y) = sum(k(x, y) for k in κ.kernels)

function kernelmatrix(κ::KernelSum, x::AbstractVector)
    return sum(kernelmatrix(k, x) for k in κ.kernels)
end

function kernelmatrix(κ::KernelSum, x::AbstractVector, y::AbstractVector)
    return sum(kernelmatrix(k, x, y) for k in κ.kernels)
end

function kernelmatrix_diag(κ::KernelSum, x::AbstractVector)
    return sum(kernelmatrix_diag(k, x) for k in κ.kernels)
end

function kernelmatrix_diag(κ::KernelSum, x::AbstractVector, y::AbstractVector)
    return sum(kernelmatrix_diag(k, x, y) for k in κ.kernels)
end

function Base.show(io::IO, κ::KernelSum)
    return printshifted(io, κ, 0)
end

function Base.:(==)(x::KernelSum, y::KernelSum)
    return (
        length(x.kernels) == length(y.kernels) &&
        all(kx == ky for (kx, ky) in zip(x.kernels, y.kernels))
    )
end

function printshifted(io::IO, κ::KernelSum, shift::Int)
    print(io, "Sum of $(length(κ)) kernels:")
    for k in κ.kernels
        print(io, "\n")
        for _ in 1:(shift + 1)
            print(io, "\t")
        end
        printshifted(io, k, shift + 2)
    end
end
