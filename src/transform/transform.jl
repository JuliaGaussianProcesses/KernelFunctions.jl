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

transform(t::ScaleTransform{<:Real},x::AbstractVecOrMat) = t.s*x
