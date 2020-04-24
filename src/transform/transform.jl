export Transform, IdentityTransform, ScaleTransform, ARDTransform, LowRankTransform, FunctionTransform, ChainTransform

include("scaletransform.jl")
include("ardtransform.jl")
include("lowranktransform.jl")
include("functiontransform.jl")
include("selecttransform.jl")
include("chaintransform.jl")

"""
    apply(t::Transform, x; obsdim::Int=defaultobs)

Apply the transform `t` vector-wise on the array `x`
"""
apply

"""
    IdentityTransform()

Return exactly the input
"""
struct IdentityTransform <: Transform end

apply(t::IdentityTransform, x::AbstractVector) = x

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
#     @adjoint transform(t::ARDTransform{<:Real},X::AbstractMatrix{<:Real},obsdim::Int) = transform(t,X,obsdim),Δ->begin
#     @show Δ,size(Δ);
#     return (obsdim == 1 ? ARD()Δ'.*X : ScaleTransform()Δ.*X,transform(t,Δ,obsdim),nothing)
#     end
#
# @adjoint transform(t::ScaleTransform{T},x::AbstractVecOrMat,obsdim::Int) where {T<:Real} = transform(t,x), Δ->(ScaleTransform(one(T)),t.s.*Δ,nothing)
