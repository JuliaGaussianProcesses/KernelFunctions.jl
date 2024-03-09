"""
    SelectTransform(dims)

Transformation that selects the dimensions `dims` of the input.

# Examples

```jldoctest
julia> dims = [1, 3, 5, 6, 7]; t = SelectTransform(dims); X = rand(100, 10);

julia> map(t, ColVecs(X)) == ColVecs(X[dims, :])
true
```
"""
struct SelectTransform{T} <: Transform
    select::T
end

set!(t::SelectTransform, dims) = t.select .= dims

duplicate(t::SelectTransform, θ) = t

(t::SelectTransform)(x::AbstractVector) = _maybe_unwrap(view(x, t.select))

_maybe_unwrap(x) = x
_maybe_unwrap(x::AbstractArray{<:Any,0}) = x[]

_map(t::SelectTransform, x::ColVecs) = _wrap(view(x.X, t.select, :), ColVecs)
_map(t::SelectTransform, x::RowVecs) = _wrap(view(x.X, :, t.select), RowVecs)

_wrap(x::AbstractVector, ::Any) = x
_wrap(X::AbstractMatrix, ::Type{T}) where {T} = T(X)

Base.show(io::IO, t::SelectTransform) = print(io, "Select Transform (dims: ", t.select, ")")
