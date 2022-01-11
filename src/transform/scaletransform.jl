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

function ScaleTransform(s::T=1.0) where {T<:Real}
    return ScaleTransform{T}([s])
end

@functor ScaleTransform

set!(t::ScaleTransform, ρ::Real) = t.s .= [ρ]

(t::ScaleTransform)(x) = only(t.s) * x

_map(t::ScaleTransform, x::AbstractVector{<:Real}) = only(t.s) .* x
_map(t::ScaleTransform, x::ColVecs) = ColVecs(only(t.s) .* x.X)
_map(t::ScaleTransform, x::RowVecs) = RowVecs(only(t.s) .* x.X)

Base.isequal(t::ScaleTransform, t2::ScaleTransform) = isequal(only(t.s), only(t2.s))

Base.show(io::IO, t::ScaleTransform) = print(io, "Scale Transform (s = ", only(t.s), ")")
