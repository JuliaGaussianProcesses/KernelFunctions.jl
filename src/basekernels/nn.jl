"""
    NeuralNetworkKernel()

Neural network kernel function.

```julia
    κ(x, y) =  asin(x' * y / sqrt[(1 + x' * x) * (1 + y' * y)])
```

"""
struct NeuralNetworkKernel <: BaseKernel end

function (κ::NeuralNetworkKernel)(x, y)
    return asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
end

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
