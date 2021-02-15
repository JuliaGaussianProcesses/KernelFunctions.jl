"""
    WienerKernel(; i::Int=0)
    WienerKernel{i}()

The `i`-times integrated Wiener process kernel function.

# Definition

For inputs ``x, x' \\in \\mathbb{R}^d``, the ``i``-times integrated Wiener process kernel
with ``i \\in \\{-1, 0, 1, 2, 3\\}`` is defined[^SDH] as
```math
k_i(x, x') = \\begin{cases}
    \\delta(x, x') & \\text{if } i=-1,\\\\
    \\min\\big(\\|x\\|_2, \\|x'\\|_2\\big) & \\text{if } i=0,\\\\
    a_{i1}^{-1} \\min\\big(\\|x\\|_2, \\|x'\\|_2\\big)^{2i + 1}
    + a_{i2}^{-1} \\|x - x'\\|_2 r_i\\big(\\|x\\|_2, \\|x'\\|_2\\big) \\min\\big(\\|x\\|_2, \\|x'\\|_2\\big)^{i + 1}
    & \\text{otherwise},
\\end{cases}
```
where the coefficients ``a`` are given by
```math
a = \\begin{bmatrix}
3 & 2 \\\\
20 & 12 \\\\
252 & 720
\\end{bmatrix}
```
and the functions ``r_i`` are defined as
```math
\\begin{aligned}
r_1(t, t') &= 1,\\\\
r_2(t, t') &= t + t' - \\frac{\\min(t, t')}{2},\\\\
r_3(t, t') &= 5 \\max(t, t')^2 + 2 tt' + 3 \\min(t, t')^2.
\\end{aligned}
```

The [`WhiteKernel`](@ref) is recovered for ``i = -1``.

[^SDH]: Schober, Duvenaud & Hennig (2014). Probabilistic ODE Solvers with Runge-Kutta Means.
"""
struct WienerKernel{I} <: Kernel
    function WienerKernel{I}() where {I}
        @check_args(WienerKernel, I, I ∈ (-1, 0, 1, 2, 3), "I ∈ {-1, 0, 1, 2, 3}")
        if I == -1
            return WhiteKernel()
        end
        return new{I}()
    end
end

function WienerKernel(; i::Integer=0)
    return WienerKernel{i}()
end

function (::WienerKernel{0})(x, y)
    X = sqrt(sum(abs2, x))
    Y = sqrt(sum(abs2, y))
    return min(X, Y)
end

function (::WienerKernel{1})(x, y)
    X = sqrt(sum(abs2, x))
    Y = sqrt(sum(abs2, y))
    minXY = min(X, Y)
    return 1 / 3 * minXY^3 + 1 / 2 * minXY^2 * euclidean(x, y)
end

function (::WienerKernel{2})(x, y)
    X = sqrt(sum(abs2, x))
    Y = sqrt(sum(abs2, y))
    minXY = min(X, Y)
    return 1 / 20 * minXY^5 + 1 / 12 * minXY^3 * euclidean(x, y) * (X + Y - 1 / 2 * minXY)
end

function (::WienerKernel{3})(x, y)
    X = sqrt(sum(abs2, x))
    Y = sqrt(sum(abs2, y))
    minXY = min(X, Y)
    return 1 / 252 * minXY^7 +
           1 / 720 * minXY^4 * euclidean(x, y) * (5 * max(X, Y)^2 + 2 * X * Y + 3 * minXY^2)
end

function Base.show(io::IO, ::WienerKernel{I}) where {I}
    return print(io, I, "-times integrated Wiener kernel")
end
