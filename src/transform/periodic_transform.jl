"""
    PeriodicTransform(f)

Makes a kernel periodic by mapping a scalar input onto the unit circle. Samples from a GP
with a kernel with this transformation applied will produce samples with frequency `f`.
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
    print(io, "Periodic Transform with frequency $(first(t.f))")
end
