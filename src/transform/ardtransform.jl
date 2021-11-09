"""
    ARDTransform(v::AbstractVector)

Transformation that multiplies the input elementwise by `v`.

# Examples

```jldoctest
julia> v = rand(10); t = ARDTransform(v); X = rand(10, 100);

julia> map(t, ColVecs(X)) == ColVecs(v .* X)
true
```
"""
struct ARDTransform{Tv<:AbstractVector{<:Real}} <: Transform
    v::Tv
end

"""
    ARDTransform(s::Real, dims::Integer)

Create an [`ARDTransform`](@ref) with vector `fill(s, dims)`.
"""
ARDTransform(s::Real, dims::Integer) = ARDTransform(fill(s, dims))

function ParameterHandling.flatten(::Type{T}, t::ARDTransform{S}) where {T<:Real,S}
    unflatten_to_ardtransform(v::Vector{T}) = ARDTransform(convert(S, map(exp, v)))
    return convert(Vector{T}, map(log, t.v)), unflatten_to_ardtransform
end

function set!(t::ARDTransform{<:AbstractVector{T}}, ρ::AbstractVector{T}) where {T<:Real}
    @assert length(ρ) == dim(t) "Trying to set a vector of size $(length(ρ)) to ARDTransform of dimension $(dim(t))"
    return t.v .= ρ
end

dim(t::ARDTransform) = length(t.v)

(t::ARDTransform)(x::Real) = first(t.v) * x
(t::ARDTransform)(x) = t.v .* x

_map(t::ARDTransform, x::AbstractVector{<:Real}) = t.v' .* x
_map(t::ARDTransform, x::ColVecs) = ColVecs(t.v .* x.X)
_map(t::ARDTransform, x::RowVecs) = RowVecs(t.v' .* x.X)

Base.isequal(t::ARDTransform, t2::ARDTransform) = isequal(t.v, t2.v)

Base.show(io::IO, t::ARDTransform) = print(io, "ARD Transform (dims: ", dim(t), ")")
