"""
    ARDTransform(v::AbstractVector)

Transformation that multiplies the input elementwise by `v`.

# Examples

```jldoctest
julia> v = rand(10); t = ARDTransform(v); X = rand(10, 100);

julia> map(t, ColVecs(X)) == ColVecs(v .* X)
true
```
"""
struct ARDTransform{Tv<:AbstractVector{<:Real}} <: Transform
    v::Tv
end

"""
    ARDTransform(s::Real, dims::Integer)

Create an [`ARDTransform`](@ref) with vector `fill(s, dims)`.
"""
ARDTransform(s::Real, dims::Integer) = ARDTransform(fill(s, dims))

@functor ARDTransform

function set!(t::ARDTransform{<:AbstractVector{T}}, ρ::AbstractVector{T}) where {T<:Real}
    @assert length(ρ) == dim(t) "Trying to set a vector of size $(length(ρ)) to ARDTransform of dimension $(dim(t))"
    return t.v .= ρ
end

dim(t::ARDTransform) = length(t.v)

(t::ARDTransform)(x::Real) = only(t.v) * x
(t::ARDTransform)(x) = t.v .* x

# Quite specific implementations required to pass correctness and performance tests.
function _map(t::ARDTransform, x::ColVecs)
    return ColVecs((t.v * ones(eltype(t.v), 1, size(x.X, 2))) .* x.X)
end
function _map(t::ARDTransform, x::RowVecs)
    return RowVecs(x.X .* (ones(eltype(t.v), size(x.X, 1)) * collect(t.v')))
end

Base.isequal(t::ARDTransform, t2::ARDTransform) = isequal(t.v, t2.v)

Base.show(io::IO, t::ARDTransform) = print(io, "ARD Transform (dims: ", dim(t), ")")
