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

# SimpleKernel interface
kappa(::ZeroKernel, ::Real) = false
metric(::ZeroKernel) = Delta()

# Optimizations
(::ZeroKernel)(x, y) = false
kernelmatrix(::ZeroKernel, x::AbstractVector) = Falses(length(x), length(x))
function kernelmatrix(::ZeroKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return Falses(length(x), length(y))
end
function kernelmatrix!(K::AbstractMatrix, ::ZeroKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return fill!(K, zero(eltype(K)))
end
function kernelmatrix!(
    K::AbstractMatrix, ::ZeroKernel, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    return fill!(K, zero(eltype(K)))
end
kernelmatrix_diag(::ZeroKernel, x::AbstractVector) = Falses(length(x))
function kernelmatrix_diag(::ZeroKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return Falses(length(x))
end
function kernelmatrix_diag!(K::AbstractVector, ::ZeroKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return fill!(K, zero(eltype(K)))
end
function kernelmatrix_diag!(
    K::AbstractVector, ::ZeroKernel, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    return fill!(K, zero(eltype(K)))
end

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

    function ConstantKernel(; c::Real=1.0, check_args::Bool=true)
        check_args && @check_args(ConstantKernel, c, c >= zero(c), "c ≥ 0")
        return new{typeof(c)}([c])
    end
end

@functor ConstantKernel

# SimpleKernel interface
kappa(κ::ConstantKernel, ::Real) = only(κ.c)
metric(::ConstantKernel) = Delta()

# Optimizations
(k::ConstantKernel)(x, y) = only(k.c)
kernelmatrix(k::ConstantKernel, x::AbstractVector) = Fill(only(k.c), length(x), length(x))
function kernelmatrix(k::ConstantKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return Fill(only(k.c), length(x), length(y))
end
function kernelmatrix!(K::AbstractMatrix, k::ConstantKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return fill!(K, only(k.c))
end
function kernelmatrix!(
    K::AbstractMatrix, k::ConstantKernel, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    return fill!(K, only(k.c))
end
kernelmatrix_diag(k::ConstantKernel, x::AbstractVector) = Fill(only(k.c), length(x))
function kernelmatrix_diag(k::ConstantKernel, x::AbstractVector, y::AbstractVector)
    validate_inputs(x, y)
    return Fill(only(k.c), length(x))
end
function kernelmatrix_diag!(K::AbstractVector, k::ConstantKernel, x::AbstractVector)
    validate_inplace_dims(K, x)
    return fill!(K, only(k.c))
end
function kernelmatrix_diag!(
    K::AbstractVector, k::ConstantKernel, x::AbstractVector, y::AbstractVector
)
    validate_inplace_dims(K, x, y)
    return fill!(K, only(k.c))
end

Base.show(io::IO, κ::ConstantKernel) = print(io, "Constant Kernel (c = ", only(κ.c), ")")
