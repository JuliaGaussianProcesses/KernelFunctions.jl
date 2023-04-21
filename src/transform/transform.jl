"""
    Transform

Abstract type defining a transformation of the input.
"""
abstract type Transform end

# We introduce our own _map for Transform so that we can work around
# https://github.com/FluxML/Zygote.jl/issues/646 and define our own pullback
# (see zygoterules.jl)
Base.map(t::Transform, x::ColVecs) = _map(t, x)
Base.map(t::Transform, x::RowVecs) = _map(t, x)

# Fallback
# No separate methods for `x::ColVecs` and `x::RowVecs` to avoid method ambiguities
function _map(t::Transform, x::AbstractVector)
    # Avoid stackoverflow
    if x isa RowVecs
        return map(t, eachrow(x.X))
    elseif x isa ColVecs
        return map(t, eachcol(x.X))
    else
        return map(t, x)
    end
end

"""
    IdentityTransform()

Transformation that returns exactly the input.
"""
struct IdentityTransform <: Transform end

(t::IdentityTransform)(x) = x

# More efficient implementation than `map(IdentityTransform(), x)`
# Introduces, however, discrepancy between `map` and `_map`
_map(::IdentityTransform, x::AbstractVector) = x

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
