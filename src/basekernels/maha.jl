"""
    MahalanobisKernel(P::AbstractMatrix)

Mahalanobis distance-based kernel given by
```math
    κ(x,y) =  exp(-r^2), r^2 = maha(x,P,y) = (x-y)'* P *(x-y)
```
where the matrix P is the metric.

"""
struct MahalanobisKernel{T<:Real, A<:AbstractMatrix{T}} <: SimpleKernel
    P::A
    function MahalanobisKernel(P::AbstractMatrix{T}) where {T<:Real}
        LinearAlgebra.checksquare(P)
        new{T,typeof(P)}(P)
    end
end

kappa(κ::MahalanobisKernel, d::T) where {T<:Real} = exp(-d)

metric(κ::MahalanobisKernel) = SqMahalanobis(κ.P)

function dot_perslice(A::AbstractMatrix, B::AbstractMatrix; dims=2)
    return reshape(sum(A .* B, dims=3-dims), :)
end

Base.show(io::IO, κ::MahalanobisKernel) = print(io, "Mahalanobis Kernel (size(P) = ", size(κ.P), ")")
