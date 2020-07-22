"""
    IndependentMOKernel(k::Kernel) <: MOKernel

A Multi-Output kernel which assumes each output is independent of the other.

"""
struct IndependentMOKernel{Tkernel<:Kernel} <: MOKernel
    kernel::Tkernel
end

function (κ::IndependentMOKernel)(x::Tuple{Vector, Int}, y::Tuple{Vector, Int})
    return κ.kernel(x[1], y[1])
end

function kernelmatrix(k::IndependentMOKernel, x::MOInput, y::MOInput)
    @assert x.out_dim == y.out_dim
    temp = k.kernel.(x.x, permutedims(y.x))
    return repeat(temp, outer=[y.out_dim, y.out_dim])
end

Base.show(io::IO, k::IndependentMOKernel) = 
print(io, string("Independent Multi-Output Kernel\n\t", string(k.kernel)))
