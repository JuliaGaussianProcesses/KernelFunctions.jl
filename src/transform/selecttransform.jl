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
    function SelectTransform{V}(dims::V) where {V}
        new{V}(dims)
    end
end

function SelectTransform(dims::V) where {V<:AbstractVector{T} where  {T<:Int}}
    @assert all(dims.>0) "Selective dimensions should all be positive integers"
    SelectTransform{V}(dims)
end

set!(t::SelectTransform{<:AbstractVector{T}},dims::AbstractVector{T}) where {T<:Int} = t.select .= dims

params(t::SelectTransform) = t.select
opt_params(t::SelectTransform) = nothing

Base.maximum(t::SelectTransform) = maximum(t.select)

function transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs)
    @boundscheck maximum(t) >= size(X,feature_dim(obsdim)) ?
        throw(DimensionMismatch("The highest index $(maximum(t)) is higher then the feature dimension of X : $(size(X,feature_dim(obsdim)))")) : nothing
    @inbounds _transform(t,X,obsdim)
end

function transform(t::SelectTransform,x::AbstractVector{<:Real},obsdim::Int=defaultobs) #TODO Add test
    @assert maximum(t) <= length(x) "The highest index $(maximum(t)) is higher then the vector length : $(length(x))"
    return @inbounds view(x,t.select)
end

_transform(t::SelectTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 2 ? view(X,t.select,:) : view(X,:,t.select)
