"""
    MahalanobisKernel(P::AbstractMatrix)

Mahalanobis distance-based kernel given by
```math
    κ(x,y) =  exp(-r^2), r^2 = maha(x,P,y) = (x-y)'*inv(P)*(x-y)
```
where the matrix P is the metric.

"""
struct MahalanobisKernel{T<:Real, A<:AbstractMatrix{T}} <: BaseKernel
    P::A
    function MahalanobisKernel(P::AbstractMatrix{T}) where {T<:Real}
        LinearAlgebra.checksquare(P)
        new{T,typeof(P)}(P)
    end
end

kappa(κ::MahalanobisKernel, d::T) where {T<:Real} = exp(-d)
metric(κ::MahalanobisKernel) = SqMahalanobis(κ.P)

Base.show(io::IO, κ::MahalanobisKernel) = print(io, "Mahalanobis Kernel (size(P) = $(size(κ.P)))")
