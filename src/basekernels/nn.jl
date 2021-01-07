"""
    NeuralNetworkKernel()

Neural network kernel function.

```math
    κ(x, y) =  asin(x' * y / sqrt[(1 + x' * x) * (1 + y' * y)])
```
# Significance
Neal (1996) pursued the limits of large models, and showed that a Bayesian neural network
becomes a Gaussian process with a **neural network kernel** as the number of units
approaches infinity. Here, we give the neural network kernel for single hidden layer
Bayesian neural network with erf (Error Function) as activation function.

# References:
- [GPML Pg 105](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf)
- [Neal(1996)](https://www.cs.toronto.edu/~radford/bnn.book.html)
- [Andrew Gordon's Thesis Pg 45](http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf)
"""
struct NeuralNetworkKernel <: Kernel end

function (κ::NeuralNetworkKernel)(x, y)
    return asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims = 1)
    Y_2 = sum(y.X .* y.X; dims = 1)
    XY = x.X' * y.X
    return asin.(XY ./ sqrt.((X_2 .+ 1)' * (Y_2 .+ 1)))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs)
    X_2_1 = sum(x.X .* x.X; dims = 1) .+ 1
    XX = x.X' * x.X
    return asin.(XX ./ sqrt.(X_2_1' * X_2_1))
end

function kernelmatrix(::NeuralNetworkKernel, x::RowVecs, y::RowVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims = 2)
    Y_2 = sum(y.X .* y.X; dims = 2)
    XY = x.X * y.X'
    return asin.(XY ./ sqrt.((X_2 .+ 1) * (Y_2 .+ 1)'))
end

function kernelmatrix(::NeuralNetworkKernel, x::RowVecs)
    X_2_1 = sum(x.X .* x.X; dims = 2) .+ 1
    XX = x.X * x.X'
    return asin.(XX ./ sqrt.(X_2_1 * X_2_1'))
end

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
