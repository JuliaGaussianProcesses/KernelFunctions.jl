"""
SelectTransform
```
    dims = [1,3,5,6,7]
    tr = SelectTransform(dims)
    X = rand(100,10)
    transform(tr,X,obsdim=2) == X[dims,:]
```
Select the dimensions `dims` that the kernel is applied to.
"""
struct SelectTransform{T<:AbstractVector{<:Int}} <: Transform
    select::T
    dim_max::Int
end

function SelectTransform(dims::AbstractVector{T}) where {T<:Int}
    @assert all(dims.>0) "Selective dimensions should all be positive integers"
    SelectTransform{T}(dims,maximum(dims))
end

function set!(t::SelectTransform{<:AbstractVector{T}},s::AbstractVector{T}) where {T<:Real}
    t.proj .= s
end

Base.maximum(t::SelectTransform) = maximum(t.select)

function transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs)
    @boundscheck t.dim_max <= size(X,feature_dim(obsdim)) ?
        throw(DimensionMismatch("The highest index $(t.dim_max) is higher then the feature dimension of X : $(size(X,feature_dim(obsdim)))")) : nothing
    @inbounds _transform(t,X,obsdim)
end

function transform(t::SelectTransform,x::AbstractVector{<:Real},obsdim::Int=defaultobs) #TODO Add test
    @assert t.dim_max <= length(x) "The highest index $(t.dim_max) is higher then the vector length : $(length(x))"
    return x[t.select]
end

_transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 2 ? X[t.select,:] : X[:,t.select]
