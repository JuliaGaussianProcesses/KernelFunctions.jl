"""
    MahaKernel(; ell::Real=1.0, p::Real=1.0)

    Mahalanobis distance-based kernel given by
```math
    κ(x,y) =  k(r^2), r^2 = maha(x,P,y) = (x-y)'*inv(P)*(x-y)
```
where the matrix P is the metric.

"""
struct MahaKernel{T<:Real} <: BaseKernel
    P
    function MahaKernel(P::AbstractMatrix{T}) where {T<:Real}
        @assert size(P)[1] == size(P)[2], "P should be a square matrix"
        new{T}(P)
    end
end

kappa(κ::MahaKernel, d::T) where {T<:Real} = d 

metric(κ::MahaKernel) = Mahalanobis(κ.P)
