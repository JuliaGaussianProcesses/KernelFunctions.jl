struct LowRankTransform{T<:AbstractMatrix{<:Real}}} <: Transform
    proj::T
end

function LowRankTransform(proj::M) where {M<:AbstractMatrix{<:Real}}
    # @check_args(LowRankTransform, proj, A > zero(T), "s > 0")
    LowRankTransform{T}(proj)
end

Base.size(tr::LowRankTransform{<:Real},i::Int) = size(tr.proj,i)
Base.size(tr::LowRankTransform{<:Real}) = size(tr.proj)

function transform(t::LowRankTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int)
    @boundscheck if size(t,2) != size(X,!Bool(obsdim-1)+1)
        throw(DimensionMismatch("The projection matrix has size $(size(t)) and cannot be used on X with dimensions $(size(X))"))
    end
    _transform(t,X,obsdim)
end
_transform(t::LowRankTransform,x::AbstractVector{<:Real}) = t.proj * x
_transform(t::LowRankTransform,X::AbstractMatrix{<:Real},obsdim::Int) = t.proj * X
