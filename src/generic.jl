## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

printshifted(io::IO, o, shift::Int) = print(io, o)

# See https://github.com/JuliaGaussianProcesses/KernelFunctions.jl/issues/96
_scale(t::ScaleTransform, metric::Euclidean, x, y) =  first(t.s) * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric::Union{SqEuclidean,DotProduct}, x, y) =  first(t.s)^2 * evaluate(metric, x, y)
_scale(t::ScaleTransform, metric, x, y) = evaluate(metric, apply(t, x), apply(t, y))

### Syntactic sugar for creating matrices and using kernel functions
function concretetypes(k, ktypes::Vector)
    isempty(subtypes(k)) ? push!(ktypes, k) : concretetypes.(subtypes(k), Ref(ktypes))
    return ktypes
end

for k in nameof.(subtypes(BaseKernel))
    @eval begin
        @deprecate($k(ρ::Real;args...),transform($k(args...),ρ))
        @deprecate($k(ρ::AbstractVector{<:Real};args...),transform($k(args...),ρ))
    end
end

# Fallback implementation of evaluate for `SimpleKernel`s.
(k::SimpleKernel)(x, y) = kappa(k, evaluate(metric(k), x, y))
