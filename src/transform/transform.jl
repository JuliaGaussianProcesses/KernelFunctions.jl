abstract type Transform end

struct TransformChain <: Transform
    transforms::Vector{Transform}
end

function TransformChain(v::AbstractVector{<:Transform})
    TransformChain(v)
end

struct InputTransform{F} <: Transform
    f::F
end

# function InputTransform(f::F) where {F}
#     InputTransform{F}(f)
# end

transform(t::InputTransform,x::T,obsdim::Int=1) where {T} = t.f(X)

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
_transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real}) = t.s .* x
_transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int) = obsdim == 1 ? t.s'.*X : t.s .* X

transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat,obsdim::Int) = transform(t,x)
transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat) = t.s .* x


struct IdentityTransform <: Transform end

transform(t::IdentityTransform,x::AbstractArray,obsdim::Int) = transform(t,x)
transform(t::IdentityTransform,x::AbstractArray) = return x

### TODO Maybe defining adjoints could help but so far it's not working


# @adjoint function ScaleTransform(s::T) where {T<:Real}
#     @check_args(ScaleTransform, s, s > zero(T), "s > 0")
#     ScaleTransform{T}(s),Δ->ScaleTransform{T}(Δ)
# end
#
# @adjoint function ScaleTransform(s::A) where {A<:AbstractVector{<:Real}}
#     @check_args(ScaleTransform, s, all(s.>zero(eltype(A))), "s > 0")
#     ScaleTransform{A}(s),Δ->begin; @show Δ,size(Δ); ScaleTransform{A}(Δ); end
# end

# @adjoint transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real}) = transform(t,x),Δ->(ScaleTransform(nothing),t.s.*Δ)
#
#     @adjoint transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int) = transform(t,X,obsdim),Δ->begin
#     @show Δ,size(Δ);
#     return (obsdim == 1 ? ScaleTransform()Δ'.*X : ScaleTransform()Δ.*X,transform(t,Δ,obsdim),nothing)
#     end
#
# @adjoint transform(t::ScaleTransform{T},x::AbstractVecOrMat,obsdim::Int) where {T<:Real} = transform(t,x), Δ->(ScaleTransform(one(T)),t.s.*Δ,nothing)
