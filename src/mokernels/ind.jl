"""
    IndependentKernel(k...) <: MOKernel

A Multi-Output kernel which assumes each output is independent of the other.
"""
struct IndependentKernel{Tkernels<:AbstractVector} <: MOKernel
    kernels::Tkernels
    function IndependentKernel(k...)
        k = collect(k)
        return new{typeof(k)}(k)
    end
end

Base.length(κ::IndependentKernel) = length(κ.kernels)


function (κ::IndependentKernel)(x::MOInput, y::MOInput)
    @assert length(κ) == x.out_dim == y.out_dim
    temp = vcat((κ.kernels[i](x.x, y.x) for i in 1:length(κ))...)
    return hcat((temp for _ in 1:length(κ))...)
end

Base.show(io::IO, ::IndependentKernel) = print(io, "Independent Multi-Output Kernel")
