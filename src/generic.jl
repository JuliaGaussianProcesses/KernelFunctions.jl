## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

# default fallback for evaluating a kernel with two arguments (such as vectors etc)
kappa(κ::Kernel, x, y) = kappa(κ, evaluate(metric(κ), x, y))
kappa(κ::TransformedKernel, x, y) = kappa(kernel(κ), apply(κ.transform,x), apply(κ.transform,y))
kappa(κ::TransformedKernel{<:Kernel,<:ScaleTransform}, x, y) = kappa(κ.kernel, _scale(κ.transform, metric(κ.kernel), x, y))
_scale(t::ScaleTransform, metric::Euclidean, x, y) =  first(t.s) * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric::Union{SqEuclidean,DotProduct}, x, y) =  first(t.s)^2 * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric, x, y) = evaluate(metric, apply(t, x), apply(t, y))

### Syntactic sugar for creating matrices and using kernel functions
for k in [:ExponentialKernel,:SqExponentialKernel,:GammaExponentialKernel,:MaternKernel,:Matern32Kernel,:Matern52Kernel,:LinearKernel,:PolynomialKernel,:ExponentiatedKernel,:ZeroKernel,:WhiteKernel,:ConstantKernel,:RationalQuadraticKernel,:GammaRationalQuadraticKernel]
    @eval begin
        @inline (κ::$k)(d::Real) = kappa(κ,d) #TODO Add test
        @inline (κ::$k)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = kappa(κ, x, y)
        @inline (κ::$k)(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, Y, obsdim=obsdim)
        @inline (κ::$k)(X::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, obsdim=obsdim)
    end
end

## Constructors for kernels without parameters
# for kernel in [:ExponentialKernel,:SqExponentialKernel,:Matern32Kernel,:Matern52Kernel,:ExponentiatedKernel]
#     @eval begin
#         $kernel() = $kernel(IdentityTransform())
#         $kernel(ρ::Real) = $kernel(ScaleTransform(ρ))
#         $kernel(ρ::AbstractVector{<:Real}) = $kernel(ARDTransform(ρ))
#     end
# end

for k in [:SqExponentialKernel,:ExponentialKernel,:GammaExponentialKernel]
    new_k = Symbol(lowercase(string(k)))
    @eval begin
        $new_k(args...) = $k(args...)
        $new_k(ρ::Real,args...) = TransformedKernel($k(args...),ScaleTransform(ρ))
        $new_k(ρ::AbstractVector{<:Real},args...) = TransformedKernel($k(args...),ARDTransform(ρ))
        $new_k(t::Transform,args...) = TransformedKernel($k(args...),t)
        export $new_k
    end
end
