"""
    ScaleTransform(l::Real)

Multiply every element of the input by `l`
```
    l = 2.0
    tr = ScaleTransform(l)
```
"""
struct ScaleTransform{T<:Real} <: Transform
    s::Vector{T}
end

function ScaleTransform(s::T=1.0) where {T<:Real}
    ScaleTransform{T}([s])
end

set!(t::ScaleTransform,ρ::Real) = t.s .= [ρ]

(t::ScaleTransform)(x) = first(t.s) * x

_map(t::ScaleTransform, x::AbstractVector{<:Real}) = t(x)
_map(t::ScaleTransform, x::ColVecs) = ColVecs(t(x.X))
_map(t::ScaleTransform, x::RowVecs) = RowVecs(t(x.X))

Base.isequal(t::ScaleTransform,t2::ScaleTransform) = isequal(first(t.s),first(t2.s))

Base.show(io::IO,t::ScaleTransform) = print(io,"Scale Transform (s = ", first(t.s), ")")
