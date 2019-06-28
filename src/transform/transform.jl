abstract type Transform{T} end

struct TransformChain{T} <: Transform{T}
end



struct InputTransform{T} <: Transform{T}

end

struct ScaleTransform{T<:Union{Real,AbstractVector{<:Real}}} <: Transform{T}
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

transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real}) = t.s.*x
transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int) = obsdim == 1 ? t.s'.*X : t.s.*X

transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat,obsdim::Int) = t.s*x

@adjoint transform(t::ScaleTransform{<:AbstractVector{<:Real}},x::AbstractVector{<:Real}) = transform(t,x),Δ->(Δ.*x,t.s.*Δ)
@adjoint transform(t::ScaleTransform{<:AbstractVector{<:Real}},X::AbstractMatrix{<:Real},obsdim::Int) = transform(t,X,obsdim),Δ->begin
@show Δ,size(Δ);
return (obsdim == 1 ? Δ'.*X : Δ.*X,transform(t,Δ,obsdim),nothing)
end

@adjoint transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat,obsdim::Int) = transform(t,x), Δ->(Δ.s.*x,t.s.*Δ)
