"""
    IndependentMOKernel(k...) <: MOKernel

A Multi-Output kernel which assumes each output is independent of the other.

"""
struct IndependentMOKernel{Tkernels<:AbstractVector} <: MOKernel
    kernels::Tkernels
end

function IndependentMOKernel(k...)
    return IndependentMOKernel(collect(k))
end

Base.length(κ::IndependentMOKernel) = length(κ.kernels)

function (κ::IndependentMOKernel)(x::MOInput, y::MOInput)
    @assert length(κ) == x.out_dim == y.out_dim
    temp = vcat((κ.kernels[i](x.x, y.x) for i in 1:length(κ))...)
    return hcat((temp for _ in 1:length(κ))...)
end

Base.show(io::IO, ::IndependentMOKernel) = print(io, "Independent Multi-Output Kernel")
