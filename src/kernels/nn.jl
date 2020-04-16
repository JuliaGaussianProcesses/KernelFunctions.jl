"""
    NeuralNetworkKernel(P::AbstractMatrix)

Neural network kernel function with a single parameter for the distance
measure. The kernel function is parameterized as:

```julia
    κ(x, y) =  asin(x' * y / sqrt[(1 + x' * x) * (1 + y' * y)])
```

"""
struct NeuralNetworkKernel <: BaseKernel end

kappa(κ::NeuralNetworkKernel, x, y) = asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
