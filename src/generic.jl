## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

# default fallback for evaluating a kernel with two arguments (such as vectors etc)
kappa(κ::Kernel, x, y) = kappa(κ, evaluate(metric(κ), x, y))
kappa(κ::TransformedKernel, x, y) = kappa(kernel(κ), apply(κ.transform,x), apply(κ.transform,y))
kappa(κ::TransformedKernel{<:BaseKernel,<:ScaleTransform}, x, y) = kappa(κ, _scale(κ.transform, metric(κ), x, y))

printshifted(io::IO, o, shift::Int) = print(io, o)
Base.show(io::IO, κ::Kernel) = print(io, nameof(typeof(κ)))

### Syntactic sugar for creating matrices and using kernel functions
function concretetypes(k, ktypes::Vector)
    isempty(subtypes(k)) ? push!(ktypes, k) : concretetypes.(subtypes(k), Ref(ktypes))
    return ktypes
end

for k in concretetypes(Kernel, [])
    @eval begin
        @inline (κ::$k)(x, y) = kappa(κ, x, y)
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
