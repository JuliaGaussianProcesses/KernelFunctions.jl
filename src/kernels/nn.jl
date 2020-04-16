"""
    NeuralNetworkKernel(P::AbstractMatrix)

    Neural network kernel function with a single parameter for the distance
    measure. The kernel function is parameterized as:
```math
    κ(x, y) =  asin(x'* P * y / sqrt[(1 + x' * P * x) * (1 + y' * P * y)])
```

"""
struct NeuralNetworkKernel{A<:AbstractMatrix{<:Real}} <: BaseKernel
    P::A
    function NeuralNetworkKernel(P::AbstractMatrix{<:Real})
        LinearAlgebra.checksquare(P)
        new{typeof(P)}(P)
    end
end

kappa(κ::NeuralNetworkKernel, x, y) = asin(x'* κ.P * y / sqrt((1 + x' * κ.P * x) * (1 + y' * κ.P * y)))

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel (dim = $(size(κ.P,2)))")
