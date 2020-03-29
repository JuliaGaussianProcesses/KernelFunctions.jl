"""
    NeuralNetOneKernel(P::AbstractMatrix)

    Neural network kernel function with a single parameter for the distance
    measure. The kernel function is parameterized as:
```math
    κ(x,y) =  asin(x'*y / sqrt((1 + x'*x)*(1 + y'*y)))
```

"""
struct NeuralNetOneKernel{T<:Real, A<:AbstractMatrix{T}} <: BaseKernel
    P::A
    function NeuralNetOneKernel(P::AbstractMatrix{T}) where {T<:Real}
        LinearAlgebra.checksquare(P)
        new{T,typeof(P)}(P)
    end
end

kappa(κ::NeuralNetOneKernel, x, y) = asin(x'*y / sqrt((1 + x'*x)*(1 + y'*y)))
)