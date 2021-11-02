"""
    PeriodicTransform(f)

Transformation that maps the input elementwise onto the unit circle with frequency `f`.

Samples from a GP with a kernel with this transformation applied to the inputs will produce
samples with frequency `f`.

# Examples

```jldoctest
julia> f = rand(); t = PeriodicTransform(f); x = rand();

julia> t(x) == [sinpi(2 * f * x), cospi(2 * f * x)]
true
```
"""
struct PeriodicTransform{T<:Real} <: Transform
    f::T
end

function ParameterHandling.flatten(::Type{T}, t::PeriodicTransform) where {T<:Real}
    f = t.f
    unflatten_to_periodictransform(v::Vector{T}) = PeriodicTransform(oftype(f, only(v)))
    return T[f], unflatten_to_periodictransform
end

dim(t::PeriodicTransform) = 2

(t::PeriodicTransform)(x::Real) = [sinpi(2 * t.f * x), cospi(2 * t.f * x)]

function _map(t::PeriodicTransform, x::AbstractVector{<:Real})
    return RowVecs(hcat(sinpi.((2 * t.f) .* x), cospi.((2 * t.f) .* x)))
end

Base.isequal(t1::PeriodicTransform, t2::PeriodicTransform) = isequal(t1.f, t2.f)

function Base.show(io::IO, t::PeriodicTransform)
    return print(io, "Periodic Transform with frequency ", t.f)
end
