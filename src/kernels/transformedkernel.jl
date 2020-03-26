"""
`TransformedKernel(k::Kernel,t::Transform)`
Return a kernel where inputs are pretransformed by `t` : `k(t(x),t(x'))`
"""
struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

function TransformedKernel(k::TransformedKernel,t::Transform)
    TransformedKernel(kernel(k),t∘k.transform)
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

transform(k::Kernel, t::Transform) = TransformedKernel(k, t)

transform(k::Kernel, ρ::Real) = TransformedKernel(k, ScaleTransform(ρ))

transform(k::Kernel,ρ::AbstractVector) = TransformedKernel(k, ARDTransform(ρ))

transform(k::BaseKernel,::Nothing) = k

kernel(κ) = κ.kernel

kappa(κ::TransformedKernel, x) = kappa(κ.kernel, x)

metric(κ::TransformedKernel) = metric(κ.kernel)

Base.show(io::IO,κ::TransformedKernel) = printshifted(io,κ,0)

function printshifted(io::IO,κ::TransformedKernel,shift::Int)
    printshifted(io,κ.kernel,shift)
    print(io,"\n"*("\t"^(shift+1))*"- $(κ.transform)")
end
