"""
    PiecewisePolynomialKernel{V}(maha::AbstractMatrix)

Piecewise Polynomial covariance function with compact support, V = 0,1,2,3.
The kernel functions are 2v times continuously differentiable and the corresponding
processes are hence v times  mean-square differentiable. The kernel function is:
```math
    κ(x, y) = max(1 - r, 0)^(j + V) * f(r, j) with j = floor(D / 2) + V + 1
```
where `r` is the Mahalanobis distance mahalanobis(x,y) with `maha` as the metric.

"""
struct PiecewisePolynomialKernel{V,A<:AbstractMatrix{<:Real}} <: SimpleKernel
    maha::A
    j::Int
    function PiecewisePolynomialKernel{V}(maha::AbstractMatrix{<:Real}) where {V}
        V in (0, 1, 2, 3) || error("Invalid paramter v=$(V). Should be 0, 1, 2 or 3.")
        LinearAlgebra.checksquare(maha)
        j = div(size(maha, 1), 2) + V + 1
        return new{V,typeof(maha)}(maha, j)
    end
end

function PiecewisePolynomialKernel(; v::Integer = 0, maha::AbstractMatrix{<:Real})
    return PiecewisePolynomialKernel{v}(maha)
end

# Have to reconstruct the type parameter
# See also https://github.com/FluxML/Functors.jl/issues/3#issuecomment-626747663
function Functors.functor(::Type{<:PiecewisePolynomialKernel{V}}, x) where {V}
    function reconstruct_kernel(xs)
        return PiecewisePolynomialKernel{V}(xs.maha)
    end
    return (maha = x.maha,), reconstruct_kernel
end

_f(κ::PiecewisePolynomialKernel{0}, r, j) = 1
_f(κ::PiecewisePolynomialKernel{1}, r, j) = 1 + (j + 1) * r
_f(κ::PiecewisePolynomialKernel{2}, r, j) = 1 + (j + 2) * r + (j^2 + 4 * j + 3) / 3 * r .^ 2
_f(κ::PiecewisePolynomialKernel{3}, r, j) =
    1 +
    (j + 3) * r +
    (6 * j^2 + 36j + 45) / 15 * r .^ 2 +
    (j^3 + 9 * j^2 + 23j + 15) / 15 * r .^ 3

kappa(κ::PiecewisePolynomialKernel{V}, r) where {V} =
    max(1 - r, 0)^(κ.j + V) * _f(κ, r, κ.j)

metric(κ::PiecewisePolynomialKernel) = Mahalanobis(κ.maha)

function Base.show(io::IO, κ::PiecewisePolynomialKernel{V}) where {V}
    print(io, "Piecewise Polynomial Kernel (v = ", V, ", size(maha) = ", size(κ.maha), ")")
end
