"""
    PiecewisePolynomialKernel(; degree::Int=0, dim::Int)
    PiecewisePolynomialKernel{degree}(dim::Int)

Piecewise polynomial kernel of degree `degree` for inputs of dimension `dim` with support in
the unit ball.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d`` of dimension ``d``, the piecewise polynomial kernel
of degree ``v \\in \\{0,1,2,3\\}`` is defined as
```math
k(x, x'; v) = \\max(1 - \\|x - x'\\|, 0)^{\\alpha(v,d)} f_{v,d}(\\|x - x'\\|),
```
where ``\\alpha(v, d) = \\lfloor \\frac{d}{2}\\rfloor + 2v + 1`` and ``f_{v,d}`` are
polynomials of degree ``v`` given by
```math
\\begin{aligned}
f_{0,d}(r) &= 1, \\\\
f_{1,d}(r) &= 1 + (j + 1) r, \\\\
f_{2,d}(r) &= 1 + (j + 2) r + ((j^2 + 4j + 3) / 3) r^2, \\\\
f_{3,d}(r) &= 1 + (j + 3) r + ((6 j^2 + 36j + 45) / 15) r^2 + ((j^3 + 9 j^2 + 23j + 15) / 15) r^3,
\\end{aligned}
```
where ``j = \\lfloor \\frac{d}{2}\\rfloor + v + 1``.

The kernel is ``2v`` times continuously differentiable and the corresponding Gaussian
process is hence ``v`` times mean-square differentiable.
"""
struct PiecewisePolynomialKernel{D,C<:Tuple} <: SimpleKernel
    alpha::Int
    coeffs::C

    function PiecewisePolynomialKernel{D}(dim::Int) where {D}
        dim > 0 || error("number of dimensions has to be positive")
        j = div(dim, 2) + D + 1
        alpha = j + D
        coeffs = piecewise_polynomial_coefficients(Val(D), j)
        return new{D,typeof(coeffs)}(alpha, coeffs)
    end
end

# TODO: remove `maha` keyword argument in next breaking release
function PiecewisePolynomialKernel(; v::Int=-1, degree::Int=v, maha=nothing, dim::Int=-1)
    if v != -1
        Base.depwarn(
            "keyword argument `v` is deprecated, use `degree` instead",
            :PiecewisePolynomialKernel,
        )
    end

    if maha !== nothing
        Base.depwarn(
            "keyword argument `maha` is deprecated, use a `LinearTransform` instead",
            :PiecewisePolynomialKernel,
        )
        dim = size(maha, 1)
        return transform(
            PiecewisePolynomialKernel{degree}(dim), LinearTransform(cholesky(maha).U)
        )
    else
        return PiecewisePolynomialKernel{degree}(dim)
    end
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

# `evalpoly` is not available on Julia < 1.4
@static if VERSION < v"1.4"
    @generated function _evalpoly(r, coeffs)
        N = length(coeffs.parameters)
        return quote
            return @evalpoly(r, $((:(coeffs[$i]) for i in 1:N)...))
        end
    end
else
    _evalpoly(r, coeffs) = evalpoly(r, coeffs)
end

kappa(κ::PiecewisePolynomialKernel, r) = max(1 - r, 0)^κ.alpha * _evalpoly(r, κ.coeffs)

metric(::PiecewisePolynomialKernel) = Euclidean()

function Base.show(io::IO, κ::PiecewisePolynomialKernel{D}) where {D}
    return print(
        io,
        "Piecewise Polynomial Kernel (degree = ",
        D,
        ", ⌊dim/2⌋ = ",
        κ.alpha - 1 - 2 * D,
        ")",
    )
end
