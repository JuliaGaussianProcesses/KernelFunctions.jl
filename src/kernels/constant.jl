"""
ZeroKernel()

Create a kernel that always returning zero
```
    κ(x,y) = 0.0
```
The output type depends of `x` and `y`
"""
struct ZeroKernel <: Kernel end

@inline kappa(κ::ZeroKernel, d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

"""
`WhiteKernel()`

```
    κ(x,y) = δ(x,y)
```
Kernel function working as an equivalent to add white noise.
"""
struct WhiteKernel <: Kernel end

@inline kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ

metric(::WhiteKernel) = Delta()

"""
`ConstantKernel(c=1.0)`
```
    κ(x,y) = c
```
Kernel function always returning a constant value `c`
"""
struct ConstantKernel{Tc<:Real} <: Kernel
    c::Tc
end

function ConstantKernel(c::Tc=1.0) where {Tc<:Real}
    ConstantKernel{Tc}(c)
end

params(k::ConstantKernel) = (k.c)
opt_params(k::ConstantKernel) = (k.c)

@inline kappa(κ::ConstantKernel,x::Real) = κ.c*one(x)

@inline metric(::ConstantKernel) = Delta()
