"""
    MahaKernel(P::AbstractMatrix, kappa::Function)

    Mahalanobis distance-based kernel given by
```math
    κ(x,y) =  kappa(r^2), r^2 = maha(x,P,y) = (x-y)'*inv(P)*(x-y)
```
where the matrix P is the metric and kappa is a user defined kernel function.

"""
struct MahaKernel{T<:Real} <: BaseKernel
    P::AbstractMatrix{T}
    kappa::Function
    function MahaKernel(P::AbstractMatrix{T}, kappa::Function) where {T<:Real}
        LinearAlgebra.checksquare(P)
        new{T}(P, kappa)
    end
end

kappa(κ::MahaKernel, d::T) where {T<:Real} = κ.kappa(d) 

metric(κ::MahaKernel) = SqMahalanobis(κ.P)
