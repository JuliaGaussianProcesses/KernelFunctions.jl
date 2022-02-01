"""
    LinearTransform(A::AbstractMatrix)

Linear transformation of the input realised by the matrix `A`.

    LinearTransform(a::AbstractVector)

Linear transformation of the input with `A = Diagonal(a)`.    

The second dimension of `A` must match the number of features of the target.

# Examples

```jldoctest
julia> A = rand(10, 5); t = LinearTransform(A); X = rand(5, 100);

julia> map(t, ColVecs(X)) == ColVecs(A * X)
true
```
"""
struct LinearTransform{T<:AbstractMatrix{<:Real}} <: Transform
    A::T
end

@functor LinearTransform

function set!(t::LinearTransform{<:AbstractMatrix{T}}, A::AbstractMatrix{T}) where {T<:Real}
    size(t.A) == size(A) || error(
        "size of the given matrix ",
        size(A),
        " and of the transformation matrix ",
        size(t.A),
        " are not the same",
    )
    return t.A .= A
end

(t::LinearTransform)(x::Real) = vec(t.A * x)
(t::LinearTransform)(x::AbstractVector{<:Real}) = t.A * x

_map(t::LinearTransform, x::AbstractVector{<:Real}) = ColVecs(t.A * collect(x'))
_map(t::LinearTransform, x::ColVecs) = ColVecs(t.A * x.X)
_map(t::LinearTransform, x::RowVecs) = RowVecs(x.X * t.A')

function Base.show(io::IO, t::LinearTransform)
    return print(io::IO, "Linear transform (size(A) = ", size(t.A), ")")
end
