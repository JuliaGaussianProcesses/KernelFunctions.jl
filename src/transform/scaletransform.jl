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

(t::ScaleTransform)(x) = first(t.s) .* x

Base.map(t::ScaleTransform, x::AbstractVector{<:Real}) = first(t.s) .* x
Base.map(t::ScaleTransform, x::ColVecs) = ColVecs(first(t.s) .* x.X)
Base.map(t::ScaleTransform, x::RowVecs) = RowVecs(first(t.s) .* x.X)

Base.isequal(t::ScaleTransform,t2::ScaleTransform) = isequal(first(t.s),first(t2.s))

Base.show(io::IO,t::ScaleTransform) = print(io,"Scale Transform (s = ", first(t.s), ")")
