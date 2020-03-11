## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

# default fallback for evaluating a kernel with two arguments (such as vectors etc)
kappa(κ::Kernel, x, y) = kappa(κ, evaluate(metric(κ), x, y))
kappa(κ::TransformedKernel, x, y) = kappa(kernel(κ), apply(κ.transform,x), apply(κ.transform,y))
kappa(κ::TransformedKernel{<:BaseKernel,<:ScaleTransform}, x, y) = kappa(κ, _scale(κ.transform, metric(κ), x, y))
_scale(t::ScaleTransform, metric::Euclidean, x, y) =  first(t.s) * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric::Union{SqEuclidean,DotProduct}, x, y) =  first(t.s)^2 * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric, x, y) = evaluate(metric, apply(t, x), apply(t, y))

printshifted(io::IO, κ::Kernel, shift::Int) = print(io, "$κ")
Base.show(io::IO, κ::Kernel) = print(io, nameof(typeof(κ)))

### Syntactic sugar for creating matrices and using kernel functions
for k in subtypes(BaseKernel)
    @eval begin
        @inline (κ::$k)(d::Real) = kappa(κ,d) #TODO Add test
        @inline (κ::$k)(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = kappa(κ, x, y)
        @inline (κ::$k)(X::AbstractMatrix{T}, Y::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, Y, obsdim=obsdim)
        @inline (κ::$k)(X::AbstractMatrix{T}; obsdim::Integer=defaultobs) where {T} = kernelmatrix(κ, X, obsdim=obsdim)
    end
end

for k in nameof.(subtypes(BaseKernel))
    @eval begin
        @deprecate($k(ρ::Real;args...),transform($k(args...),ρ))
        @deprecate($k(ρ::AbstractVector{<:Real};args...),transform($k(args...),ρ))
    end
end
