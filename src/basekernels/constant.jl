"""
    ZeroKernel()

Create a kernel that always returning zero
```
    κ(x,y) = 0.0
```
The output type depends of `x` and `y`
"""
struct ZeroKernel <: SimpleKernel end

kappa(κ::ZeroKernel, d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

Base.show(io::IO, ::ZeroKernel) = print(io, "Zero Kernel")

"""
    WhiteKernel()

```
    κ(x,y) = δ(x,y)
```
Kernel function working as an equivalent to add white noise. Can also be called via `EyeKernel()`
"""
struct WhiteKernel <: SimpleKernel end

"""
    EyeKernel()

See [`WhiteKernel`](@ref)
"""
const EyeKernel = WhiteKernel

kappa(κ::WhiteKernel, δₓₓ::Real) = δₓₓ

metric(::WhiteKernel) = Delta()

Base.show(io::IO, ::WhiteKernel) = print(io, "White Kernel")

"""
    ConstantKernel(; c=1.0)

Kernel function always returning a constant value `c`
```
    κ(x,y) = c
```
"""
struct ConstantKernel{Tc<:Real} <: SimpleKernel
    c::Vector{Tc}
    function ConstantKernel(; c::T=1.0) where {T<:Real}
        return new{T}([c])
    end
end

@functor ConstantKernel

kappa(κ::ConstantKernel, x::Real) = first(κ.c) * one(x)

metric(::ConstantKernel) = Delta()

Base.show(io::IO, κ::ConstantKernel) = print(io, "Constant Kernel (c = ", first(κ.c), ")")
