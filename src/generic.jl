@inline metric(κ::Kernel) = κ.metric

## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

# default fallback for evaluating a kernel with two arguments (such as vectors etc)
kappa(κ::Kernel, x, y) = kappa(κ, evaluate(metric(κ), transform(κ, x), transform(κ, y)))

### Syntactic sugar for creating matrices and using kernel functions
for k in [:ExponentialKernel,:SqExponentialKernel,:GammaExponentialKernel,:MaternKernel,:Matern32Kernel,:Matern52Kernel,:LinearKernel,:PolynomialKernel,:ExponentiatedKernel,:ZeroKernel,:WhiteKernel,:ConstantKernel,:RationalQuadraticKernel,:GammaRationalQuadraticKernel]
    @eval begin
        @inline (κ::$k)(d::Real) = kappa(κ,d) #TODO Add test
        @inline (κ::$k)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = kappa(κ, x, y)
        @inline (κ::$k)(X::AbstractMatrix{T},Y::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,X,Y,obsdim=obsdim)
        @inline (κ::$k)(X::AbstractMatrix{T};obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ,X,obsdim=obsdim)
    end
end

### Transform generics
@inline transform(κ::Kernel) = κ.transform
@inline transform(κ::Kernel, x) = transform(transform(κ), x)
@inline transform(κ::Kernel, x, obsdim::Int) = transform(transform(κ), x, obsdim)

## Constructors for kernels without parameters
for kernel in [:ExponentialKernel,:SqExponentialKernel,:Matern32Kernel,:Matern52Kernel,:ExponentiatedKernel]
    @eval begin
        $kernel(ρ::Real=1.0) = $kernel(ScaleTransform(ρ))
        $kernel(ρ::AbstractVector{<:Real}) = $kernel(ARDTransform(ρ))
    end
end
