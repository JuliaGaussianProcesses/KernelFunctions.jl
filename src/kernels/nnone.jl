"""
    NeuralNetOneKernel(P::AbstractMatrix)

    Neural network covariance function with a single parameter for the distance
    measure. The covariance function is parameterized as:
```math
    κ(x,y) =  asin(x'*z / sqrt((1 + x'*x)*(1 + z'*z)))
```

"""
struct NeuralNetOneKernel{T<:Real, A<:AbstractMatrix{T}} <: BaseKernel
    P::A
    function NeuralNetOneKernel(P::AbstractMatrix{T}) where {T<:Real}
        LinearAlgebra.checksquare(P)
        new{T,typeof(P)}(P)
    end
end

kappa(κ::NeuralNetOneKernel, d::T) where {T<:Real} = exp(-d) 

metric(κ::NeuralNetOneKernel) = SqMahalanobis(κ.P)
