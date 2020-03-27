"""
    TransformedKernel(k::Kernel,t::Transform)
Return a kernel where inputs are pretransformed by `t` : `k(t(x),t(x'))`
Can also be called via [transform](@ref) : `transform(k, t)`
"""
struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

"""
```julia
    transform(k::BaseKernel, t::Transform) (1)
    transform(k::BaseKernel, ρ::Real) (2)
    transform(k::BaseKernel, ρ::AbstractVector) (3)
```
(1) Create a TransformedKernel with transform `t` and kernel `k`
(2) Same as (1) with a `ScaleTransform` with scale `ρ`
(3) Same as (1) with an `ARDTransform` with scales `ρ`
"""
transform

transform(k::BaseKernel, t::Transform) = TransformedKernel(k, t)

transform(k::BaseKernel, ρ::Real) = TransformedKernel(k, ScaleTransform(ρ))

transform(k::BaseKernel,ρ::AbstractVector) = TransformedKernel(k, ARDTransform(ρ))

kernel(κ) = κ.kernel

kappa(κ::TransformedKernel, x) = kappa(κ.kernel, x)

metric(κ::TransformedKernel) = metric(κ.kernel)

Base.show(io::IO, κ::TransformedKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::TransformedKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    print(io,"\n" * ("\t" ^ (shift + 1)) * "- $(κ.transform)")
end
