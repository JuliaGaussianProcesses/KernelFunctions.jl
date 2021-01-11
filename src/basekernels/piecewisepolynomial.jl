"""
    PiecewisePolynomialKernel(; v::Int=0, d::Int)
    PiecewisePolynomialKernel{v}(d::Int)

Piecewise polynomial kernel with compact support.

The kernel is defined for ``x, x' \\in \\mathbb{R}^d`` and ``v \\in \\{0,1,2,3\\}`` as
```math
k(x, x'; v) = \\max(1 - \\|x - x'\\|, 0)^{j + v} f_v(\\|x - x'\\|, j),
```
where ``j = \\lfloor \\frac{d}{2}\\rfloor + v + 1``, and ``f_v`` are polynomials defined as
follows:
```math
\\begin{aligned}
f_0(r, j) &= 1, \\\\
f_1(r, j) &= 1 + (j + 1) r, \\\\
f_2(r, j) &= 1 + (j + 2) r + ((j^2 + 4j + 3) / 3) r^2, \\\\
f_3(r, j) &= 1 + (j + 3) r + ((6 j^2 + 36j + 45) / 15) r^2 + ((j^3 + 9 j^2 + 23j + 15) / 15) r^3.
\\end{aligned}
```

The kernel is ``2v`` times continuously differentiable and the corresponding Gaussian
process is hence ``v`` times mean-square differentiable.
"""
struct PiecewisePolynomialKernel{V} <: SimpleKernel
    j::Int

    function PiecewisePolynomialKernel{V}(d::Int) where {V}
        V in (0, 1, 2, 3) || error("Invalid parameter V=$(V). Should be 0, 1, 2 or 3.")
        d > 0 || error("number of dimensions has to be positive")
        j = div(d, 2) + V + 1
        return new{V}(j)
    end
end

# TODO: remove `maha` keyword argument in next breaking release
function PiecewisePolynomialKernel(; v::Int=0, maha=nothing, d::Int=-1)
    if maha !== nothing
        Base.depwarn("keyword argument `maha` is deprecated", :PiecewisePolynomialKernel)
        d = size(maha, 1)
        return transform(PiecewisePolynomialKernel{v}(d), LinearTransform(cholesky(maha).U))
    else
        return PiecewisePolynomialKernel{v}(d)
    end
end

_f(::PiecewisePolynomialKernel{1}, r, j) = 1 + (j + 1) * r
_f(::PiecewisePolynomialKernel{2}, r, j) = 1 + (j + 2) * r + (j^2 + 4 * j + 3) / 3 * r^2
function _f(::PiecewisePolynomialKernel{3}, r, j)
    return 1 +
        (j + 3) * r +
        (6 * j^2 + 36j + 45) / 15 * r ^ 2 +
        (j^3 + 9 * j^2 + 23j + 15) / 15 * r ^ 3
end

kappa(κ::PiecewisePolynomialKernel{0}, r) = max(1 - r, 0)^κ.j
function kappa(κ::PiecewisePolynomialKernel{V}, r) where {V}
    return max(1 - r, 0)^(κ.j + V) * _f(κ, r, κ.j)
end

metric(::PiecewisePolynomialKernel) = Euclidean()

function Base.show(io::IO, κ::PiecewisePolynomialKernel{V}) where {V}
    return print(io, "Piecewise Polynomial Kernel (v = ", V, ", ⌊d/2⌋ = ", κ.j - 1 - V, ")")
end
