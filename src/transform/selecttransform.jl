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
end

function SelectTransform(dims::V) where {V<:AbstractVector{T} where  {T<:Int}}
    @assert all(dims.>0) "Selective dimensions should all be positive integers"
    SelectTransform{V}(dims)
end

get_params(t::SelectTransform) = t.select
get_params(k::Kernel{T,<:SelectTransform}) where {T} = get_params(k.transform)

set!(t::SelectTransform{<:AbstractVector{T}},dims::AbstractVector{T}) where {T<:Int} = t.select .= dims
set_params!(k::Kernel{T,<:SelectTransform{Td}},dims::AbstractVector{Td}) where {T,Td<:Int} = set!(k.transform,dims)

Base.maximum(t::SelectTransform) = maximum(t.select)

function transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs)
    @boundscheck maximum(t) >= size(X,feature_dim(obsdim)) ?
        throw(DimensionMismatch("The highest index $(maximum(t)) is higher then the feature dimension of X : $(size(X,feature_dim(obsdim)))")) : nothing
    @inbounds _transform(t,X,obsdim)
end

function transform(t::SelectTransform,x::AbstractVector{<:Real},obsdim::Int=defaultobs) #TODO Add test
    @assert maximum(t) <= length(x) "The highest index $(maximum(t)) is higher then the vector length : $(length(x))"
    return @inbounds x[t.select]
end

_transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 2 ? X[t.select,:] : X[:,t.select]
