@inline metric(κ::Kernel) = κ.metric

## Allows to iterate over kernels
Base.length(::Kernel) = 1 #TODO Add test

Base.iterate(k::Kernel) = (k,nothing) #TODO Add test
Base.iterate(k::Kernel, ::Any) = nothing #TODO Add test

### Syntactic sugar for creating matrices and using kernel functions
for k in [:ExponentialKernel,:SqExponentialKernel,:GammaExponentialKernel,:MaternKernel,:Matern32Kernel,:Matern52Kernel,:LinearKernel,:PolynomialKernel,:ExponentiatedKernel,:ZeroKernel,:WhiteKernel,:ConstantKernel,:RationalQuadraticKernel,:GammaRationalQuadraticKernel]
    @eval begin
        @inline (κ::$k)(d::Real) = kappa(κ,d) #TODO Add test
        @inline (κ::$k)(x::AbstractVector{<:Real},y::AbstractVector{<:Real}) = kappa(κ,evaluate(κ.metric,transform(κ,x),transform(κ,y)))
        @inline (κ::$k)(X::AbstractMatrix{T},Y::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,X,Y,obsdim=obsdim)
        @inline (κ::$k)(X::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,X,obsdim=obsdim)
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
