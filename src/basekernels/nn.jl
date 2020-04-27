"""
    NeuralNetworkKernel()

Neural network kernel function.

```math
    κ(x, y) =  asin(x' * y / sqrt[(1 + x' * x) * (1 + y' * y)])
```
# Significance
Neal (1996) pursued the limits of large models, and showed that a Bayesian neural network becomes a Gaussian process with a **neural network kernel** as the number of units approaches infinity. Here we give the neural network kernel for single hidden layer Bayesian neural network with erf (Error Function) as activation function

# References:
- [Gaussian Processes for Machine Learning Pg 105](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf)
- [Neal(1996)](https://www.cs.toronto.edu/~radford/bnn.book.html)
- [Andrew Gordon's Thesis Pg 45](http://www.cs.cmu.edu/~andrewgw/andrewgwthesis.pdf)
"""
struct NeuralNetworkKernel <: BaseKernel end

function (κ::NeuralNetworkKernel)(x, y)
    return asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
end

Base.show(io::IO, κ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
