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
    s::T

    function ScaleTransform(s::Real)
        @check_args(ScaleTransform, s > zero(s), "s > 0")
        return new{typeof(s)}(s)
    end
end

ScaleTransform() = ScaleTransform(1.0)



function ParameterHandling.flatten(::Type{T}, t::ScaleTransform{S}) where {T<:Real,S<:Real}
    s = t.s
    function unflatten_to_scaletransform(v::Vector{T})
        length(v) == 1 || error("incorrect number of parameters")
        ScaleTransform(S(first(v)))
    end
    return T[s], unflatten_to_scaletransform
end

(t::ScaleTransform)(x) = t.s * x

_map(t::ScaleTransform, x::AbstractVector{<:Real}) = t.s .* x
_map(t::ScaleTransform, x::ColVecs) = ColVecs(t.s .* x.X)
_map(t::ScaleTransform, x::RowVecs) = RowVecs(t.s .* x.X)

Base.isequal(t::ScaleTransform, t2::ScaleTransform) = isequal(t.s, t2.s)

Base.show(io::IO, t::ScaleTransform) = print(io, "Scale Transform (s = ", t.s, ")")
