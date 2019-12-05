"""
LowRankTransform
```
    P = rand(10,5)
    tr = LowRankTransform(P)
```
Apply the low-rank projection realised by the matrix `P`
The second dimension of `P` must match the number of features of the target.
"""
struct LowRankTransform{T<:AbstractMatrix{<:Real}} <: Transform
    proj::T
end

function set!(t::LowRankTransform{<:AbstractMatrix{T}},M::AbstractMatrix{T}) where {T<:Real}
    @assert size(t) == size(M) "Size of the given matrix $(size(M)) and the projection matrix $(size(t)) are not the same"
    t.proj .= M
end

params(t::LowRankTransform) = t.proj

Base.size(tr::LowRankTransform,i::Int) = size(tr.proj,i)
Base.size(tr::LowRankTransform) = size(tr.proj) #  TODO Add test

function transform(t::LowRankTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs)
    @boundscheck size(t,2) != size(X,feature_dim(obsdim)) ?
        throw(DimensionMismatch("The projection matrix has size $(size(t)) and cannot be used on X with dimensions $(size(X))")) : nothing
    @inbounds _transform(t,X,obsdim)
end

function transform(t::LowRankTransform,x::AbstractVector{<:Real},obsdim::Int=defaultobs) #TODO Add test
    @assert size(t,2) == length(x) "Vector has wrong dimensions $(length(x)) compared to projection matrix"
    t.proj*x
end

_transform(t::LowRankTransform,X::AbstractVecOrMat{<:Real},obsdim::Int=defaultobs) = obsdim == 2 ? t.proj * X : X * t.proj'
