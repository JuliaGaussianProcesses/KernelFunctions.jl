"""
    SelectTransform(dims::AbstractVector{Int})

Select the dimensions `dims` that the kernel is applied to.
```
    dims = [1,3,5,6,7]
    tr = SelectTransform(dims)
    X = rand(100,10)
    transform(tr,X,obsdim=2) == X[dims,:]
```
"""
struct SelectTransform{T<:AbstractVector{Int}} <: Transform
    select::T
    function SelectTransform{V}(dims::V) where {V<:AbstractVector{Int}}
        @assert all(dims .> 0) "Selective dimensions should all be positive integers"
        return new{V}(dims)
    end
end

SelectTransform(x::T) where {T<:AbstractVector{Int}} = SelectTransform{T}(x)

set!(t::SelectTransform{<:AbstractVector{T}}, dims::AbstractVector{T}) where {T<:Int} = t.select .= dims

duplicate(t::SelectTransform,θ) = t

(t::SelectTransform)(x::AbstractVector) = view(x, t.select)

_map(t::SelectTransform, x::ColVecs) = ColVecs(view(x.X, t.select, :))
_map(t::SelectTransform, x::RowVecs) = RowVecs(view(x.X, :, t.select))

Base.show(io::IO, t::SelectTransform) = print(io, "Select Transform (dims: ", t.select, ")")
