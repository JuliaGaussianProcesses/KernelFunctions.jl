"""
    MaternKernel(; ν = 1.0)

The matern kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = 2^{1-ν}/Γ(ν)*(√(2ν)‖x-y‖)^ν K_ν(√(2ν)‖x-y‖)
```
For `ν=n+1/2, n=0,1,2,...` it can be simplified and you should instead use 
[`ExponentialKernel`](@ref) for `n=0`, [`Matern32Kernel`](@ref), for `n=1`, 
[`Matern52Kernel`](@ref) for `n=2` and [`SqExponentialKernel`](@ref) for `n=∞`.
"""
struct MaternKernel{Tν<:Real} <: SimpleKernel
    ν::Vector{Tν}
    function MaternKernel(;nu::T=1.5, ν::T=nu) where {T<:Real}
        @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
        return new{T}([ν])
    end
end

@functor MaternKernel

@inline function kappa(κ::MaternKernel, d::Real)
    result = _matern(first(κ.ν), d)
    return ifelse(iszero(d), one(result), result)
end

function _matern(ν::Real, d::Real)
    y = sqrt(2ν) * d
    return exp((one(d) - ν) * logtwo - loggamma(ν) + ν * log(y) + log(besselk(ν, y)))
end

metric(::MaternKernel) = Euclidean()

Base.show(io::IO, κ::MaternKernel) = print(io, "Matern Kernel (ν = ", first(κ.ν), ")")

## Matern12Kernel = ExponentialKernel aliased in exponential.jl

"""
    Matern32Kernel()

The matern 3/2 kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = (1+√(3)‖x-y‖)exp(-√(3)‖x-y‖)
```
"""
struct Matern32Kernel <: SimpleKernel end

kappa(κ::Matern32Kernel, d::Real) = (1 + sqrt(3) * d) * exp(-sqrt(3) * d)

metric(::Matern32Kernel) = Euclidean()

Base.show(io::IO, ::Matern32Kernel) = print(io, "Matern 3/2 Kernel")

"""
    Matern52Kernel()

The matern 5/2 kernel is a Mercer kernel given by the formula:
```
    κ(x,y) = (1+√(5)‖x-y‖ + 5/3‖x-y‖^2)exp(-√(5)‖x-y‖)
```
"""
struct Matern52Kernel <: SimpleKernel end

kappa(κ::Matern52Kernel, d::Real) = (1 + sqrt(5) * d + 5 * d^2 / 3) * exp(-sqrt(5) * d)

metric(::Matern52Kernel) = Euclidean()

Base.show(io::IO, ::Matern52Kernel) = print(io, "Matern 5/2 Kernel")
