@inline metric(κ::Kernel) = κ.metric

### Syntactic sugar for creating matrices and using kernel functions
for k in [:ExponentialKernel,:SqExponentialKernel,:GammaExponentialKernel,:MaternKernel,:Matern32Kernel,:Matern52Kernel,:LinearKernel,:PolynomialKernel,:ExponentiatedKernel,:ZeroKernel,:WhiteKernel,:ConstantKernel,:RationalQuadraticKernel,:GammaRationalQuadraticKernel]
    @eval begin
        @inline (κ::$k)(d::Real) = kappa(κ,d)
        @inline (κ::$k)(x::AbstractVector{T},y::AbstractVector{T}) where {T} = kernel(κ,evaluate(κ.(metric),x,y))
        @inline (κ::$k)(x::AbstractMatrix{T},y::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,x,y,obsdim=obsdim)
        @inline (κ::$k)(x::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,x,obsdim=obsdim)
    end
end

### Transform generics
@inline transform(κ::Kernel) = κ.transform
@inline transform(κ::Kernel,x::AbstractVecOrMat) = transform(κ.transform,x)
@inline transform(κ::Kernel,x::AbstractVecOrMat,obsdim::Int) = transform(κ.transform,x,obsdim)

## Constructors for kernels without parameters
for kernel in [:ExponentialKernel,:SqExponentialKernel,:Matern32Kernel,:Matern52Kernel,:ExponentiatedKernel]
    @eval begin
        $kernel(ρ::T=1.0) where {T<:Real} =   $kernel{T,ScaleTransform{T}}(ScaleTransform(ρ))
        $kernel(ρ::A) where {A<:AbstractVector{<:Real}} = $kernel{eltype(A),ScaleTransform{A}}(ScaleTransform(ρ))
        $kernel(t::Tr) where {Tr<:Transform} = $kernel{eltype(t),Tr}(t)
    end
end
