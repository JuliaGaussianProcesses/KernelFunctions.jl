"""
    ARDTransform(v::AbstractVector)
    ARDTransform(s::Real, dims::Int)

Multiply every vector of observation by `v` element-wise
```
    v = rand(3)
    tr = ARDTransform(v)
```
"""
struct ARDTransform{Tv<:AbstractVector{<:Real}} <: Transform
    v::Tv
end

ARDTransform(s::Real, dims::Integer) = ARDTransform(fill(s, dims))

function set!(t::ARDTransform{<:AbstractVector{T}}, ρ::AbstractVector{T}) where {T<:Real}
    @assert length(ρ) == dim(t) "Trying to set a vector of size $(length(ρ)) to ARDTransform of dimension $(dim(t))"
    t.v .= ρ
end

dim(t::ARDTransform) = length(t.v)

(t::ARDTransform)(x::Real) = first(t.v) * x
(t::ARDTransform)(x) = t.v .* x

Base.map(t::ARDTransform, x::AbstractVector{<:Real}) = t.v' .* x
_map(t::ARDTransform, x::ColVecs) = ColVecs(t.v .* x.X)
_map(t::ARDTransform, x::RowVecs) = RowVecs(t.v' .* x.X)

Base.isequal(t::ARDTransform, t2::ARDTransform) = isequal(t.v, t2.v)

Base.show(io::IO, t::ARDTransform) =
    print(io, "ARD Transform (dims: ", dim(t),")")
