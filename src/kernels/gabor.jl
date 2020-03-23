"""
    GaborKernel(; ell::Real=1.0, p::Real=1.0)

Gabor kernel with length scale ell and period p. Given by
```math
    κ(x,y) =  h(x-z), h(t) = exp(-sum(t.^2./(2*ell.^2)))*cos(2*pi*sum(t./p))
```

"""
struct GaborKernel{T<:Real} <: BaseKernel
    ell::T
    p::T
    function GaborKernel(;ell::T=1.0, p::T=1.0) where {T<:Real}
        new{T}(ell, p)
    end
end

kappa(κ::GaborKernel, d::T) where {T<:Real} = exp(-sum(d.^2 ./(2*κ.ell.^2)))*cospi(2*sum(d)./ κ.p)

metric(::GaborKernel) = Euclidean()
