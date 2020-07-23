"""
    IndependentMOKernel(k::Kernel) <: MOKernel

A Multi-Output kernel which assumes each output is independent of the other.
"""
struct IndependentMOKernel{Tkernel<:Kernel} <: MOKernel
    kernel::Tkernel
end

function (κ::IndependentMOKernel)(x::Tuple{Vector, Int}, y::Tuple{Vector, Int})
    if last(x) == last(y)
        return κ.kernel(first(x), first(y))
    else
        return 0.0
    end
end

function kernelmatrix(k::IndependentMOKernel, x::MOInput, y::MOInput)
    @assert x.out_dim == y.out_dim
    temp = k.kernel.(x.x, permutedims(y.x))
    return cat((temp for _ in 1:y.out_dim)...; dims=(1,2))
end

function Base.show(io::IO, k::IndependentMOKernel)
    print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
end
