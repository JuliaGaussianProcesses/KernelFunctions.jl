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

function ParameterHandling.flatten(::Type{T}, k::KernelProduct) where {T<:Real}
    vecs_and_backs = map(Base.Fix1(flatten, T), k.kernels)
    vecs = map(first, vecs_and_backs)
    length_vecs = map(length, vecs)
    backs = map(last, vecs_and_backs)
    flat_vecs = reduce(vcat, vecs)
    function unflatten_to_kernelproduct(v::Vector{T})
        length(v) == length(flat_vecs) || error("incorrect number of parameters")
        offset = Ref(0)
        kernels = map(backs, length_vecs) do back, length_vec
            oldoffset = offset[]
            newoffset = offset[] = oldoffset + length_vec
            return back(v[(oldoffset + 1):newoffset])
        end
        return KernelProduct(kernels)
    end
    return flat_vecs, unflatten_to_kernelproduct
end

Base.length(k::KernelProduct) = length(k.kernels)

function _hadamard(f, ks::Tuple, args...)
    return f(first(ks), args...) .* _hadamard(f, Base.tail(ks), args...)
end
_hadamard(f, ks::Tuple{Tx}, args...) where {Tx} = f(only(ks), args...)

(κ::KernelProduct)(x, y) = _hadamard((k, x, y) -> k(x, y), κ.kernels, x, y)

function kernelmatrix(κ::KernelProduct, x::AbstractVector)
    return _hadamard(kernelmatrix, κ.kernels, x)
end

function kernelmatrix(κ::KernelProduct, x::AbstractVector, y::AbstractVector)
    return _hadamard(kernelmatrix, κ.kernels, x, y)
end

function kernelmatrix_diag(κ::KernelProduct, x::AbstractVector)
    return _hadamard(kernelmatrix_diag, κ.kernels, x)
end

function kernelmatrix_diag(κ::KernelProduct, x::AbstractVector, y::AbstractVector)
    return _hadamard(kernelmatrix_diag, κ.kernels, x, y)
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
