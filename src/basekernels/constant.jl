"""
    ZeroKernel()

Zero kernel.

# Definition

For inputs ``x, x'``, the zero kernel is defined as
```math
k(x, x') = 0.
```
The output type depends on ``x`` and ``x'``.

See also: [`ConstantKernel`](@ref)
"""
struct ZeroKernel <: SimpleKernel end

kappa(κ::ZeroKernel, d::T) where {T<:Real} = zero(T)

metric(::ZeroKernel) = Delta()

Base.show(io::IO, ::ZeroKernel) = print(io, "Zero Kernel")

"""
    WhiteKernel()

White noise kernel.

# Definition

For inputs ``x, x'``, the white noise kernel is defined as
```math
k(x, x') = \\delta(x, x').
```
"""
struct WhiteKernel <: SimpleKernel end

"""
    EyeKernel()

Alias of [`WhiteKernel`](@ref).
"""
const EyeKernel = WhiteKernel

kappa(κ::WhiteKernel, δₓₓ::Real) = δₓₓ

metric(::WhiteKernel) = Delta()

Base.show(io::IO, ::WhiteKernel) = print(io, "White Kernel")

"""
    ConstantKernel(; c::Real=1.0)

Kernel of constant value `c`.

# Definition

For inputs ``x, x'``, the kernel of constant value ``c \\geq 0`` is defined as
```math
k(x, x') = c.
```

See also: [`ZeroKernel`](@ref)
"""
struct ConstantKernel{Tc<:Real} <: SimpleKernel
    c::Vector{Tc}

    function ConstantKernel(; c::Real=1.0)
        @check_args(ConstantKernel, c, c >= zero(c), "c ≥ 0")
        return new{typeof(c)}([c])
    end
end

@functor ConstantKernel

kappa(κ::ConstantKernel, x::Real) = only(κ.c) * one(x)

metric(::ConstantKernel) = Delta()

function kernelmatrix(k::ConstantKernel, x::AbstractVector)
    return Fill(only(k.c), length(x), length(x))
end

function kernelmatrix(k::ConstantKernel, x::AbstractVector, y::AbstractVector)
    return Fill(only(k.c), length(x), length(y))
end

Base.show(io::IO, κ::ConstantKernel) = print(io, "Constant Kernel (c = ", only(κ.c), ")")
