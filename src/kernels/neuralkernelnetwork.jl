export LinearLayer, product, Primitive, NeuralKernelNetwork

# Linear layer, perform linear transformation to input array
# x₁ = softplus.(W) * x₀
struct LinearLayer{T,MT<:AbstractArray{T}}
    W::MT
end
@functor LinearLayer
LinearLayer(in_dim, out_dim) = LinearLayer(randn(out_dim, in_dim))
(lin::LinearLayer)(x) = softplus.(lin.W) * x

function Base.show(io::IO, layer::LinearLayer)
    return print(io, "LinearLayer(", size(layer.W, 2), ", ", size(layer.W, 1), ")")
end

# Product function, given an 2d array whose size is M×N, product layer will
# multiply every m neighboring rows of the array elementwisely to obtain
# an new array of size (M÷m)×N
function product(x, step=2)
    m, n = size(x)
    m % step == 0 || error("the first dimension of inputs must be multiple of step")
    new_x = reshape(x, step, m ÷ step, n)
    return .*([new_x[i, :, :] for i in 1:step]...)
end

# Primitive layer, mainly act as a container to hold basic kernels for the neural kernel network
struct Primitive{T}
    kernels::T
    Primitive(ks...) = new{typeof(ks)}(ks)
end
@functor Primitive

# flatten k kernel matrices of size Mk×Nk, and concatenate these 1d array into a k×(Mk*Nk) 2d array
_cat_kernel_array(x) = vcat([reshape(x[i], 1, :) for i in 1:length(x)]...)

# NOTE, though we implement `ew` & `pw` function for Primitive, it isn't a subtype of Kernel
# type, I do this because it will facilitate writing NeuralKernelNetwork
ew(p::Primitive, x) = _cat_kernel_array(map(k -> kernelmatrix_diag(k, x), p.kernels))
pw(p::Primitive, x) = _cat_kernel_array(map(k -> kernelmatrix(k, x), p.kernels))

function ew(p::Primitive, x, x′)
    return _cat_kernel_array(map(k -> kernelmatrix_diag(k, x, x′), p.kernels))
end
pw(p::Primitive, x, x′) = _cat_kernel_array(map(k -> kernelmatrix(k, x, x′), p.kernels))

function Base.show(io::IO, layer::Primitive)
    print(io, "Primitive(")
    join(io, layer.kernels, ", ")
    return print(io, ")")
end

"""
    NeuralKernelNetwork(primitives, nn)

Constructs a Neural Kernel Network (NKN) [1].

`primitives` are the based kernels, combined by `nn`.

```julia
k1 = 0.6 * (SEKernel() ∘ ScaleTransform(0.5))
k2 = 0.4 * (Matern32Kernel() ∘ ScaleTransform(0.1))
primitives = Primitive(k1, k2)
nkn = NeuralKernelNetwork(primitives, Chain(LinearLayer(2, 2), product))
```

[1] - Sun, Shengyang, et al. "Differentiable compositional kernel learning for Gaussian
    processes." International Conference on Machine Learning. PMLR, 2018.
"""
struct NeuralKernelNetwork{PT,NNT} <: Kernel
    primitives::PT
    nn::NNT
end
@functor NeuralKernelNetwork

# use this function to reshape the 1d array back to kernel matrix
_rebuild_kernel(x, n, m) = reshape(x, n, m)
_rebuild_diag(x) = reshape(x, :)

(κ::NeuralKernelNetwork)(x, y) = only(kernelmatrix(κ, [x], [y]))

function kernelmatrix_diag(nkn::NeuralKernelNetwork, x::AbstractVector)
    return _rebuild_diag(nkn.nn(ew(nkn.primitives, x)))
end

function kernelmatrix(nkn::NeuralKernelNetwork, x::AbstractVector)
    return _rebuild_kernel(nkn.nn(pw(nkn.primitives, x)), length(x), length(x))
end

function kernelmatrix_diag(nkn::NeuralKernelNetwork, x::AbstractVector, x′::AbstractVector)
    return _rebuild_diag(nkn.nn(ew(nkn.primitives, x, x′)))
end

function kernelmatrix(nkn::NeuralKernelNetwork, x::AbstractVector, x′::AbstractVector)
    return _rebuild_kernel(nkn.nn(pw(nkn.primitives, x, x′)), length(x), length(x′))
end

function kernelmatrix_diag!(K::AbstractVector, nkn::NeuralKernelNetwork, x::AbstractVector)
    K .= kernelmatrix_diag(nkn, x)
    return K
end

function kernelmatrix!(K::AbstractMatrix, nkn::NeuralKernelNetwork, x::AbstractVector)
    K .= kernelmatrix(nkn, x)
    return K
end

function kernelmatrix_diag!(
    K::AbstractVector, nkn::NeuralKernelNetwork, x::AbstractVector, x′::AbstractVector,
)
    K .= kernelmatrix_diag(nkn, x, x′)
    return K
end

function kernelmatrix!(
    K::AbstractMatrix, nkn::NeuralKernelNetwork, x::AbstractVector, x′::AbstractVector
)
    K .= kernelmatrix(nkn, x, x′)
    return K
end

function Base.show(io::IO, kernel::NeuralKernelNetwork)
    print(io, "NeuralKernelNetwork(")
    join(io, [kernel.primitives, kernel.nn], ", ")
    print(io, ")")
end
