## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

printshifted(io::IO, o, shift::Int) = print(io, o)

dims_are_compatible(::Kernel, x) = true

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

# This is type piracy. We should not doing this.
function Distances.pairwise(d::PreMetric, x::AbstractVector{<:Real})
    return pairwise(d, reshape(x, :, 1); dims=1)
end

function Distances.pairwise(
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return pairwise(d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end

function Distances.pairwise!(out::AbstractMatrix, d::PreMetric, x::AbstractVector{<:Real})
    return pairwise!(out, d, reshape(x, :, 1); dims=1)
end

function Distances.pairwise!(
    out::AbstractMatrix,
    d::PreMetric,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    return pairwise!(out, d, reshape(x, :, 1), reshape(y, :, 1); dims=1)
end
