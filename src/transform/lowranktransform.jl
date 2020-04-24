"""
    LowRankTransform(P::AbstractMatrix)

Apply the low-rank projection realised by the matrix `P`
The second dimension of `P` must match the number of features of the target.
```
    P = rand(10,5)
    tr = LowRankTransform(P)
```
"""
struct LowRankTransform{T<:AbstractMatrix{<:Real}} <: Transform
    proj::T
end

function set!(t::LowRankTransform{<:AbstractMatrix{T}},M::AbstractMatrix{T}) where {T<:Real}
    @assert size(t) == size(M) "Size of the given matrix $(size(M)) and the projection matrix $(size(t)) are not the same"
    t.proj .= M
end


Base.size(tr::LowRankTransform,i::Int) = size(tr.proj,i)
Base.size(tr::LowRankTransform) = size(tr.proj) #  TODO Add test

(t::LowRankTransform)(x::Real) = t([x])
(t::LowRankTransform)(x::AbstractVector{<:Real}) = t.proj * x

function Base.map(t::LowRankTransform, x::AbstractVector{<:Real})
    return ColVecs(t.proj * reshape(x, 1, :))
end
Base.map(t::LowRankTransform, x::ColVecs) = ColVecs(t.proj * x.X)
Base.map(t::LowRankTransform, x::RowVecs) = RowVecs(x.X * t.proj')

Base.show(io::IO, t::LowRankTransform) = print(io::IO, "Low Rank Transform (size(P) = ", size(t.proj), ")")
