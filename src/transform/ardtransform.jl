"""
ARD Transform
```
    v = rand(3)
    tr = ARDTransform(v)
```
Multiply every vector of observation by `v` element-wise
"""
struct ARDTransform{T,N} <: Transform
    v::Vector{T}
end

function ARDTransform(s::T,dims::Integer) where {T<:Real}
    @check_args(ARDTransform, s, s > zero(T), "s > 0")
    ARDTransform{T,dims}(fill(s,dims))
end

function ARDTransform(v::AbstractVector{T}) where {T<:Real}
    @check_args(ARDTransform, v, all(v.>zero(T)), "v > 0")
    ARDTransform{T,length(v)}(v)
end

function set!(t::ARDTransform{T},ρ::AbstractVector{T}) where {T<:Real}
    @assert length(ρ) == dim(t) "Trying to set a vector of size $(length(ρ)) to ARDTransform of dimension $(dim(t))"
    t.v .= ρ
end

params(t::ARDTransform) = t.v
dim(t::ARDTransform) = length(t.v)

function apply(t::ARDTransform,X::AbstractMatrix{<:Real};obsdim::Int)
    @boundscheck if dim(t) != size(X,feature_dim(obsdim))
        throw(DimensionMismatch("Array has size $(size(X,!Bool(obsdim-1)+1)) on dimension $(!Bool(obsdim-1)+1)) which does not match the length of the scale transform length , $(dim(t)).")) #TODO Add test
    end
    _transform(t,X,obsdim)
end
apply(t::ARDTransform,x::AbstractVector{<:Real};obsdim::Int=defaultobs) = t.v .* x
_transform(t::ARDTransform,X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 1 ? t.v'.*X : t.v .* X

Base.isequal(t::ARDTransform,t2::ARDTransform) = isequal(t.v,t2.v)
