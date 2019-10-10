"""
    Scale Transform
"""
struct ScaleTransform{T<:Union{Real,AbstractVector{<:Real}}} <: Transform
    s::T
end

function ScaleTransform(s::T=1.0) where {T<:Real}
    @check_args(ScaleTransform, s, s > zero(T), "s > 0")
    ScaleTransform{T}(s)
end

function ScaleTransform(s::T,dims::Integer) where {T<:Real}
    @check_args(ScaleTransform, s, s > zero(T), "s > 0")
    ScaleTransform{Vector{T}}(fill(s,dims))
end

function ScaleTransform(s::A) where {A<:AbstractVector{<:Real}}
    @check_args(ScaleTransform, s, all(s.>zero(eltype(A))), "s > 0")
    ScaleTransform{A}(s)
end

dim(str::ScaleTransform{<:Real}) = 1
dim(str::ScaleTransform{<:AbstractVector{<:Real}}) = length(str.s)

function transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int)
    @boundscheck if dim(t) != size(X,!Bool(obsdim-1)+1)
        throw(DimensionMismatch("Array has size $(size(X,!Bool(obsdim-1)+1)) on dimension $(!Bool(obsdim-1)+1)) which does not match the length of the scale transform length , $(dim(t))."))
    end
    _transform(t,X,obsdim)
end
transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real},obsdim::Int=defaultobs) = t.s .* x
_transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 1 ? t.s'.*X : t.s .* X

transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat,obsdim::Int=defaultobs) = t.s .* x
