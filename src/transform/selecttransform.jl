"""
    SelectTransform(dims::Union{AbstractVector{Int}, AbstractVector{Symbol}})

Select the dimensions `dims` that the kernel is applied to. `dims` can be either all
integers or symbols.
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

_map(t::SelectTransform, x::ColVecs) = ColVecs(view(x.X, t.select, :))
_map(t::SelectTransform, x::RowVecs) = RowVecs(view(x.X, :, t.select))

Base.show(io::IO, t::SelectTransform) = print(io, "Select Transform (dims: ", t.select, ")")
