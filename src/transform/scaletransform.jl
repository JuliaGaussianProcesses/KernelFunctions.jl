"""
Scale Transform
```
    l = 2.0
    tr = ScaleTransform(l)
    v = rand(3)
    tr = ScaleTransform(v)
```
Multiply every element of the matrix by `l` for a scalar
Multiply every vector of observation by `v` element-wise for a vector
"""
struct ScaleTransform{T<:Union{Base.RefValue{<:Real},AbstractVector{<:Real}}} <: Transform
    s::T
end

function ScaleTransform(s::T=1.0) where {T<:Real}
    @check_args(ScaleTransform, s, s > zero(T), "s > 0")
    ScaleTransform{Base.RefValue{T}}(Ref(s))
end

function ScaleTransform(s::T,dims::Integer) where {T<:Real} # TODO Add test
    @check_args(ScaleTransform, s, s > zero(T), "s > 0")
    ScaleTransform{Vector{T}}(fill(s,dims))
end

function ScaleTransform(s::A) where {A<:AbstractVector{<:Real}}
    @check_args(ScaleTransform, s, all(s.>zero(eltype(A))), "s > 0")
    ScaleTransform{A}(s)
end

function set!(t::ScaleTransform{Base.RefValue{T}},ρ::T) where {T<:Real}
    t.s[] = ρ
end

function set!(t::ScaleTransform{AbstractVector{T}},ρ::AbstractVector{T}) where {T<:Real}
    @assert length(ρ) == dim(t) "Trying to set a vector of size $(length(ρ)) to ScaleTransform of dimension $(dim(t))"
    t.s .= ρ
end

dim(str::ScaleTransform{Base.RefValue{<:Real}}) = 1 #TODO Add test
dim(str::ScaleTransform{<:AbstractVector{<:Real}}) = length(str.s)

function transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int)
    @boundscheck if dim(t) != size(X,!Bool(obsdim-1)+1)
        throw(DimensionMismatch("Array has size $(size(X,!Bool(obsdim-1)+1)) on dimension $(!Bool(obsdim-1)+1)) which does not match the length of the scale transform length , $(dim(t)).")) #TODO Add test
    end
    _transform(t,X,obsdim)
end
transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real},obsdim::Int=defaultobs) = t.s .* x
_transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int=defaultobs) = obsdim == 1 ? t.s'.*X : t.s .* X

transform(t::ScaleTransform{<:Base.RefValue{<:Real}},x::AbstractVecOrMat,obsdim::Int=defaultobs) = t.s[] .* x

Base.isequal(t::ScaleTransform{T},t2::ScaleTransform{T}) where {T<:Base.RefValue{<:Real}} = isequal(t.s[],t2.s[])
Base.isequal(t::ScaleTransform{T},t2::ScaleTransform{T}) where {T<:AbstractVector{<:Real}} = isequal(t.s,t2.s)
