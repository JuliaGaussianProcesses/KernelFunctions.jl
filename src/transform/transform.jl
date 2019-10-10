export Transform, ScaleTransform, LowRankTransform, FunctionTransform, TransformChain


abstract type Transform end

include("scaletransform.jl")
include("lowranktransform.jl")
include("functiontransform.jl")

struct TransformChain <: Transform
    transforms::Vector{Transform}
end

Base.length(t::TransformChain) = length(t.transforms)

function TransformChain(v::AbstractVector{<:Transform})
    TransformChain(v)
end

function transform(t::TransformChain,X::T,obsdim::Int=defaultobs) where {T}
    Xtr = copy(X)
    for tr in t.transforms
        Xtr = transform(tr,Xtr,obsdim)
    end
    return Xtr
end

Base.:∘(t₁::Transform,t₂::Transform) = TransformChain([t₂,t₁])
Base.:∘(t::Transform,tc::TransformChain) = TransformChain(vcat(tc.transforms,t))
Base.:∘(tc::TransformChain,t::Transform) = TransformChain(vcat(t,tc.transforms))

struct IdentityTransform <: Transform end

transform(t::IdentityTransform,x::AbstractArray,obsdim::Int=defaultobs) = x

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
