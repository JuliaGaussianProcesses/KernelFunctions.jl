"""
    NeuralNetworkKernel()

Kernel of a Gaussian process obtained as the limit of a Bayesian neural network with a
single hidden layer as the number of units goes to infinity.

# Definition

Consider the single-layer Bayesian neural network
``f \\colon \\mathbb{R}^d \\to \\mathbb{R}`` with ``h`` hidden units defined by
```math
f(x; b, v, u) = b + \\sqrt{\\frac{\\pi}{2}} \\sum_{i=1}^{h} v_i \\mathrm{erf}\\big(u_i^\\top x\\big),
```
where ``\\mathrm{erf}`` is the error function, and with prior distributions
```math
\\begin{aligned}
b &\\sim \\mathcal{N}(0, \\sigma_b^2),\\\\
v &\\sim \\mathcal{N}(0, \\sigma_v^2 \\mathrm{I}_{h}/h),\\\\
u_i &\\sim \\mathcal{N}(0, \\mathrm{I}_{d}/2) \\qquad (i = 1,\\ldots,h).
\\end{aligned}
```
As ``h \\to \\infty``, the neural network converges to the Gaussian process
```math
g(\\cdot) \\sim \\mathcal{GP}\\big(0, \\sigma_b^2 + \\sigma_v^2 k(\\cdot, \\cdot)\\big),
```
where the neural network kernel ``k`` is given by
```math
k(x, x') = \\arcsin\\left(\\frac{x^\\top x'}{\\sqrt{\\big(1 + \\|x\\|^2_2\\big) \\big(1 + \\|x'\\|_2^2\\big)}}\\right)
```
for inputs ``x, x' \\in \\mathbb{R}^d``.[^CW]

[^CW]: C. K. I. Williams (1998). Computation with infinite neural networks.
"""
struct NeuralNetworkKernel <: Kernel end

@noparams NeuralNetworkKernel

function (κ::NeuralNetworkKernel)(x, y)
    return asin(dot(x, y) / sqrt((1 + sum(abs2, x)) * (1 + sum(abs2, y))))
end

function kernelmatrix(
    k::NeuralNetworkKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return kernelmatrix(k, _to_colvecs(x), _to_colvecs(y))
end

function kernelmatrix(k::NeuralNetworkKernel, x::AbstractVector{<:Real})
    return kernelmatrix(k, _to_colvecs(x))
end

function kernelmatrix_diag(
    k::NeuralNetworkKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    return kernelmatrix_diag(k, _to_colvecs(x), _to_colvecs(y))
end

function kernelmatrix_diag(k::NeuralNetworkKernel, x::AbstractVector{<:Real})
    return kernelmatrix_diag(k, _to_colvecs(x))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims=1)
    Y_2 = sum(y.X .* y.X; dims=1)
    XY = x.X' * y.X
    return asin.(XY ./ sqrt.((X_2 .+ 1)' * (Y_2 .+ 1)))
end

function kernelmatrix(::NeuralNetworkKernel, x::ColVecs)
    X_2_1 = sum(x.X .* x.X; dims=1) .+ 1
    XX = x.X' * x.X
    return asin.(XX ./ sqrt.(X_2_1' * X_2_1))
end

function kernelmatrix_diag(::NeuralNetworkKernel, x::ColVecs)
    x_2 = vec(sum(x.X .* x.X; dims=1))
    return asin.(x_2 ./ (x_2 .+ 1))
end

function kernelmatrix_diag(::NeuralNetworkKernel, x::ColVecs, y::ColVecs)
    validate_inputs(x, y)
    x_2 = vec(sum(x.X .* x.X; dims=1) .+ 1)
    y_2 = vec(sum(y.X .* y.X; dims=1) .+ 1)
    xy = vec(sum(x.X' .* y.X'; dims=2))
    return asin.(xy ./ sqrt.(x_2 .* y_2))
end

function kernelmatrix(::NeuralNetworkKernel, x::RowVecs, y::RowVecs)
    validate_inputs(x, y)
    X_2 = sum(x.X .* x.X; dims=2)
    Y_2 = sum(y.X .* y.X; dims=2)
    XY = x.X * y.X'
    return asin.(XY ./ sqrt.((X_2 .+ 1) * (Y_2 .+ 1)'))
end

function kernelmatrix(::NeuralNetworkKernel, x::RowVecs)
    X_2_1 = sum(x.X .* x.X; dims=2) .+ 1
    XX = x.X * x.X'
    return asin.(XX ./ sqrt.(X_2_1 * X_2_1'))
end

function kernelmatrix_diag(::NeuralNetworkKernel, x::RowVecs)
    x_2 = vec(sum(x.X .* x.X; dims=2))
    return asin.(x_2 ./ (x_2 .+ 1))
end

function kernelmatrix_diag(::NeuralNetworkKernel, x::RowVecs, y::RowVecs)
    validate_inputs(x, y)
    x_2 = vec(sum(x.X .* x.X; dims=2) .+ 1)
    y_2 = vec(sum(y.X .* y.X; dims=2) .+ 1)
    xy = vec(sum(x.X .* y.X; dims=2))
    return asin.(xy ./ sqrt.(x_2 .* y_2))
end

Base.show(io::IO, ::NeuralNetworkKernel) = print(io, "Neural Network Kernel")
