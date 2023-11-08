"""
    ScaleTransform(l::Real)

Transformation that multiplies the input elementwise with `l`.

# Examples

```jldoctest
julia> l = rand(); t = ScaleTransform(l); X = rand(100, 10);

julia> map(t, ColVecs(X)) == ColVecs(l .* X)
true
```
"""
struct ScaleTransform{T<:Real} <: Transform
    s::T

    function ScaleTransform(s::Real)
        @check_args(ScaleTransform, s, s > zero(s), "s > 0")
        return new{typeof(s)}(s)
    end
end

ScaleTransform() = ScaleTransform(1.0)

function ParameterHandling.flatten(::Type{T}, t::ScaleTransform{S}) where {T<:Real,S<:Real}
    unflatten_to_scaletransform(v::Vector{T}) = ScaleTransform(S(exp(only(v))))
    return T[log(t.s)], unflatten_to_scaletransform
end

(t::ScaleTransform)(x) = t.s * x

_map(t::ScaleTransform, x::AbstractVector{<:Real}) = t.s .* x
_map(t::ScaleTransform, x::ColVecs) = ColVecs(t.s .* x.X)
_map(t::ScaleTransform, x::RowVecs) = RowVecs(t.s .* x.X)

Base.isequal(t::ScaleTransform, t2::ScaleTransform) = isequal(t.s, t2.s)

Base.show(io::IO, t::ScaleTransform) = print(io, "Scale Transform (s = ", t.s, ")")

# Helpers

"""
    median_heuristic_transform(distance, x::AbstractVector)

Create a [`ScaleTransform`](@ref) that divides the input elementwise by the median
`distance` of the data points in `x`.

The `distance` has to support pairwise evaluation with `KernelFunctions.pairwise`. All
`PreMetric`s of the package [Distances.jl](https://github.com/JuliaStats/Distances.jl) such
as `Euclidean` satisfy this requirement automatically.

# Examples

```jldoctest
julia> using Distances, Statistics

julia> x = ColVecs(rand(100, 10));

julia> t = median_heuristic_transform(Euclidean(), x);

julia> y = map(t, x);

julia> median(euclidean(y[i], y[j]) for i in 1:10, j in 1:10 if i != j) ≈ 1
true
```
"""
function median_heuristic_transform(f, x::AbstractVector)
    # Compute pairwise distances between **different** elements
    n = length(x)
    distances = vec(pairwise(f, x))
    deleteat!(distances, 1:(n + 1):(n^2))

    return ScaleTransform(inv(median!(distances)))
end
