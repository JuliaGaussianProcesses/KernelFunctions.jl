"""
    ScaleTransform(l::Real)

Transformation that multiplies the input elementwise with `l`.

# Examples

```jldoctest
julia> l = rand(); t = ScaleTransform(l); X = rand(100, 10);

julia> map(t, ColVecs(X)) == ColVecs(l .* X)
true
```
"""
struct ScaleTransform{T<:Real} <: Transform
    s::Vector{T}
end
Base.==(a::ScaleTransform, b::ScaleTransform) = a.s == b.s

function ScaleTransform(s::T=1.0) where {T<:Real}
    return ScaleTransform{T}([s])
end

@functor ScaleTransform

set!(t::ScaleTransform, ρ::Real) = t.s .= [ρ]

(t::ScaleTransform)(x) = first(t.s) * x

_map(t::ScaleTransform, x::AbstractVector{<:Real}) = first(t.s) .* x
_map(t::ScaleTransform, x::ColVecs) = ColVecs(first(t.s) .* x.X)
_map(t::ScaleTransform, x::RowVecs) = RowVecs(first(t.s) .* x.X)

Base.isequal(t::ScaleTransform, t2::ScaleTransform) = isequal(first(t.s), first(t2.s))

Base.show(io::IO, t::ScaleTransform) = print(io, "Scale Transform (s = ", first(t.s), ")")
