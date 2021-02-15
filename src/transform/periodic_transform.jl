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
struct PeriodicTransform{Tf<:AbstractVector{<:Real}} <: Transform
    f::Tf
end

@functor PeriodicTransform

PeriodicTransform(f::Real) = PeriodicTransform([f])

dim(t::PeriodicTransform) = 2

(t::PeriodicTransform)(x::Real) = [sinpi(2 * first(t.f) * x), cospi(2 * first(t.f) * x)]

function _map(t::PeriodicTransform, x::AbstractVector{<:Real})
    return RowVecs(hcat(sinpi.((2 * first(t.f)) .* x), cospi.((2 * first(t.f)) .* x)))
end

function Base.isequal(t1::PeriodicTransform, t2::PeriodicTransform)
    return isequal(first(t1.f), first(t2.f))
end

function Base.show(io::IO, t::PeriodicTransform)
    return print(io, "Periodic Transform with frequency $(first(t.f))")
end
