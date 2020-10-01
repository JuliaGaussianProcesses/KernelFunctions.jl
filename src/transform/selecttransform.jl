"""
    SelectTransform(dims)

Select the dimensions `dims` that the kernel is applied to.
```
    dims = [1,3,5,6,7]
    tr = SelectTransform(dims)
    X = rand(100,10)
    transform(tr,X,obsdim=2) == X[dims,:]
```
"""
struct SelectTransform{T} <: Transform
    select::T
end

set!(t::SelectTransform, dims) = t.select .= dims

duplicate(t::SelectTransform,θ) = t

(t::SelectTransform)(x::AbstractVector) = view(x, t.select)

_map(t::SelectTransform, x::ColVecs) = _wrap(view(x.X, t.select, :), typeof(x))
_map(t::SelectTransform, x::RowVecs) = _wrap(view(x.X, :, t.select), typeof(x))

_wrap(x::AbstractVector{<:Real}, ::Any) = x
_wrap(X::AbstractMatrix{<:Real}, ::Type{T}) where {T} = T(X)

Base.show(io::IO, t::SelectTransform) = print(io, "Select Transform (dims: ", t.select, ")")
