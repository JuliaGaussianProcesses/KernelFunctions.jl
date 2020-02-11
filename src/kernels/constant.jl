"""
ZeroKernel()

Create a kernel that always returning zero
```
    κ(x,y) = 0.0
```
The output type depends of `x` and `y`
"""
struct ZeroKernel <: BaseKernel end

kappa(κ::ZeroKernel, d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

"""
`WhiteKernel()`

```
    κ(x,y) = δ(x,y)
```
Kernel function working as an equivalent to add white noise.
"""
struct WhiteKernel <: BaseKernel end

kappa(κ::WhiteKernel,δₓₓ::Real) = δₓₓ

metric(::WhiteKernel) = Delta()

"""
`ConstantKernel(c=1.0)`
```
    κ(x,y) = c
```
Kernel function always returning a constant value `c`
"""
struct ConstantKernel{Tc<:Real} <: BaseKernel
    c::Tc
    function ConstantKernel(c::T=1.0) where {T<:Real}
        new{T}(c)
    end
end

params(k::ConstantKernel) = (k.c,)
opt_params(k::ConstantKernel) = (k.c,)

kappa(κ::ConstantKernel,x::Real) = κ.c*one(x)

metric(::ConstantKernel) = Delta()
