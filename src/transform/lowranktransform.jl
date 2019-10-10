struct LowRankTransform{T<:AbstractMatrix{<:Real}} <: Transform
    proj::T
end

Base.size(tr::LowRankTransform,i::Int) = size(tr.proj,i)
Base.size(tr::LowRankTransform) = size(tr.proj)

function transform(t::LowRankTransform,X::AbstractMatrix{<:Real},obsdim::Int)
    @boundscheck size(t,2) != size(X,feature_dim(obsdim)) ?
        throw(DimensionMismatch("The projection matrix has size $(size(t)) and cannot be used on X with dimensions $(size(X))")) : nothing
    @inbounds _transform(t,X,obsdim)
end
function transform(t::LowRankTransform,x::AbstractVector{<:Real})
    @assert size(t,2) == length(x) "Vector has wrong dimensions"
    t.proj*X
end

_transform(t::LowRankTransform,X::AbstractVecOrMat{<:Real},obsdim::Int) = obsdim == 2 ? t.proj * X : X * t.proj'
