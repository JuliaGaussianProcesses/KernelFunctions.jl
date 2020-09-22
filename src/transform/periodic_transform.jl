"""
    PeriodicTransform{Tf<:Ref{<:Real}} <: Transform

Makes a kernel periodic by transforming an input to be periodic.
"""
struct PeriodicTransform{Tf<:Ref{<:Real}} <: Transform
    f::Tf
end

@functor PeriodicTransform

PeriodicTransform(f::Real) = PeriodicTransform(Ref(f))

dim(t::PeriodicTransform) = 2

(t::PeriodicTransform)(x::Real) = [cosp((2 * t.f[]) * x), sinpi((2 * t.f[]) * x)]

function _map(t::PeriodicTransform, x::AbstractVector{<:Real})
    return RowVecs(hcat(cospi.((2 * t.f[]) .* x), sinpi.((2 * t.f[]) .* x)))
end

Base.isequal(t1::PeriodicTransform, t2::PeriodicTransform) = isequal(t1.f[], t2.f[])

function Base.show(io::IO, t::PeriodicTransform)
    print(io, "Periodic Transform with frequency $(t.f[])")
end
