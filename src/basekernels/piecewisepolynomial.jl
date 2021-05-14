@doc raw"""
    PiecewisePolynomialKernel(; dim::Int, degree::Int=0, metric=Euclidean())
    PiecewisePolynomialKernel{degree}(; dim::Int, metric=Euclidean())

Piecewise polynomial kernel of degree `degree` for inputs of dimension `dim` with support in
the unit ball with respect to the `metric`.

# Definition

For inputs ``x, x'`` of dimension ``m`` and metric ``d(\\cdot, \\cdot)``, the piecewise
polynomial kernel of degree ``v \in \{0,1,2,3\}`` is defined as
```math
k(x, x'; v) = \max(1 - d(x, x'), 0)^{\alpha(v,m)} f_{v,m}(d(x, x')),
```
where ``\alpha(v, m) = \lfloor \frac{m}{2}\rfloor + 2v + 1`` and ``f_{v,m}`` are
polynomials of degree ``v`` given by
```math
\begin{aligned}
f_{0,m}(r) &= 1, \\
f_{1,m}(r) &= 1 + (j + 1) r, \\
f_{2,m}(r) &= 1 + (j + 2) r + \big((j^2 + 4j + 3) / 3\big) r^2, \\
f_{3,m}(r) &= 1 + (j + 3) r + \big((6 j^2 + 36j + 45) / 15\big) r^2 + \big((j^3 + 9 j^2 + 23j + 15) / 15\big) r^3,
\end{aligned}
```
where ``j = \lfloor \frac{m}{2}\rfloor + v + 1``.
By default, ``d`` is the Euclidean metric ``d(x, x') = \|x - x'\|_2``.

The kernel is ``2v`` times continuously differentiable and the corresponding Gaussian
process is hence ``v`` times mean-square differentiable.
"""
struct PiecewisePolynomialKernel{D,C<:Tuple,M} <: SimpleKernel
    alpha::Int
    coeffs::C
    metric::M

    function PiecewisePolynomialKernel{D}(; dim::Int, metric=Euclidean()) where {D}
        dim > 0 || error("number of dimensions has to be positive")
        j = div(dim, 2) + D + 1
        alpha = j + D
        coeffs = piecewise_polynomial_coefficients(Val(D), j)
        return new{D,typeof(coeffs),typeof(metric)}(alpha, coeffs, metric)
    end
end

function PiecewisePolynomialKernel(; degree::Int=0, kwargs...)
    return PiecewisePolynomialKernel{degree}(; kwargs...)
end

piecewise_polynomial_coefficients(::Val{0}, ::Int) = (1,)
piecewise_polynomial_coefficients(::Val{1}, j::Int) = (1, j + 1)
piecewise_polynomial_coefficients(::Val{2}, j::Int) = (1, j + 2, (j^2 + 4 * j)//3 + 1)
function piecewise_polynomial_coefficients(::Val{3}, j::Int)
    return (1, j + 3, (2 * j^2 + 12 * j)//5 + 3, (j^3 + 9 * j^2 + 23 * j)//15 + 1)
end
function piecewise_polynomial_coefficients(::Val{D}, ::Int) where {D}
    return error("invalid degree $D, only 0, 1, 2, or 3 are supported")
end

kappa(κ::PiecewisePolynomialKernel, r) = max(1 - r, 0)^κ.alpha * evalpoly(r, κ.coeffs)

metric(k::PiecewisePolynomialKernel) = k.metric

function Base.show(io::IO, κ::PiecewisePolynomialKernel{D}) where {D}
    return print(
        io,
        "Piecewise Polynomial Kernel (degree = ",
        D,
        ", ⌊dim/2⌋ = ",
        κ.alpha - 1 - 2 * D,
        ", metric = ",
        κ.metric,
        ")",
    )
end
