"""
    MaternKernel([ρ=1.0,[ν=1.0]])

The matern kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = 2^{1-ν}/Γ(ν)*(√(2ν)‖x-y‖)^ν K_ν(√(2ν)‖x-y‖)
```

For `ν=n+1/2, n=0,1,2,...` it can be simplified and you should instead use `ExponentialKernel` for `n=0`, `Matern32Kernel`, for `n=1`, Matern52Kernel for `n=2` and `SqExponentialKernel` for `n=∞`.
`ρ` is the lengthscale parameter(s) or the transform object.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> MaternKernel()
MaternKernel{Float64,Float64}(1.0,1.0)

julia> MaternKernel(2.0f0,3.0)
MaternKernel{Float32,Float32}(2.0,3.0)

julia> MaternKernel([2.0,3.0],2.5)
MaternKernel{Float64,Array{Float64}}([2.0,3.0],2.5)
```
"""
struct MaternKernel{T,Tr,Tν<:Real} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    ν::Real
    function MaternKernel{T,Tr}(transform::Tr,ν::Real) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean(),ν)
    end
end

function MaternKernel(ρ::T₁=1.0,ν::T₂=1.5) where {T₁<:Real,T₂<:Real}
    @check_args(MaternKernel, ν, ν > zero(T₂), "ν > 0")
    MaternKernel{T₁,ScaleTransform{T₁},T₂}(ScaleTransform(ρ),ν)
end

function MaternKernel(ρ::A,ν::T=1.5) where {A<:AbstractVector{<:Real},T<:Real}
    @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
    MaternKernel{eltype(A),ScaleTransform{A},T}(ScaleTransform(ρ),ν)
end

function MaternKernel(t::Tr,ν::T=1.5) where {Tr<:Transform,T<:Real}
    @check_args(MaternKernel, ν, ν > zero(T), "ν > 0")
    MaternKernel{eltype(t),Tr,T}(t,ν)
end

@inline kappa(κ::MaternKernel, d::Real) where {T} = exp((1.0-κ.ν)*logtwo - (logabsgamma(κ.ν))[1] - κ.ν*log(sqrt(2κ.ν)*d))*besselk(κ.ν,sqrt(2κ.ν)*d)

"""
    Matern32Kernel(ρ=1.0)

The matern 3/2 kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = (1+√(3)‖x-y‖)exp(√(3)‖x-y‖)
```

`ρ` is the lengthscale parameter(s) or a transform object.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> Matern32Kernel()
Matern32Kernel{Float64,Float64}(1.0)

julia> Matern32Kernel(2.0f0)
Matern32Kernel{Float32,Float32}(2.0)

julia> Matern32Kernel([2.0,3.0],2.5)
Matern32Kernel{Float64,Array{Float64}}([2.0,3.0])
```
"""
struct Matern32Kernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function Matern32Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern32Kernel, d::T) where {T<:Real} = (1+sqrt(3)*d)*exp(-sqrt(3)*d)

"""
    Matern52Kernel(ρ=1.0)

The matern 5/2 kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = (1+√(5)‖x-y‖ + 5‖x-y‖^2/3)exp(√(5)‖x-y‖)
```

`ρ` is the lengthscale parameter(s) or a transform object.

# Examples

```jldoctest; setup = :(using KernelFunctions)
julia> Matern52Kernel()
Matern52Kernel{Float64,Float64}(1.0)

julia> Matern52Kernel(2.0f0)
Matern52Kernel{Float32,Float32}(2.0)

julia> Matern52Kernel([2.0,3.0],2.5)
Matern52Kernel{Float64,Array{Float64}}([2.0,3.0])
```
"""
struct Matern52Kernel{T,Tr} <: Kernel{T,Tr}
    transform::Tr
    metric::Euclidean
    function Matern52Kernel{T,Tr}(transform::Tr) where {T,Tr<:Transform}
        return new{T,Tr}(transform,Euclidean())
    end
end

@inline kappa(κ::Matern52Kernel, d::Real) where {T} = (1+sqrt(5)*d+5*d^2/3)*exp(-sqrt(5)*d)
