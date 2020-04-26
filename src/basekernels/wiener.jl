"""
    WienerKernel{i}()

i-times integrated Wiener process kernel function.

- For i=-1, this is just the white noise covariance, see [`WhiteKernel`](@ref).
- For i= 0, this is the Wiener process covariance,
- For i= 1, this is the integrated Wiener process covariance (velocity),
- For i= 2, this is the twice-integrated Wiener process covariance (accel.),
- For i= 3, this is the thrice-integrated Wiener process covariance,

where `κᵢ` is given by

```math
    κ₋₁(x, y) =  δ(x, y)
    i >= 0, κᵢ(x, y) = 1 / ai * min(x, y)^(2i + 1) + bi * min(x, y)^(i + 1) * |x - y| * ri(x, y),
    with the coefficients ai, bi and the residual ri(x, y) defined as follows:
        i = 0, ai =   1, bi = 0
        i = 1, ai =   3, bi = 1/  2, ri(x, y) = 1
        i = 2, ai =  20, bi = 1/ 12, ri(x, y) = x + y - 1 / 2 * min(x, y)
        i = 3, ai = 252, bi = 1/720, ri(x, y) = 5 * max(x, y)² + 2 * x * z + 3 * min(x, y)²
```

# References:
See the paper *Probabilistic ODE Solvers with Runge-Kutta Means* by Schober, Duvenaud and Hennig, NIPS, 2014, for more details.

"""
struct WienerKernel{I} <: BaseKernel
    function WienerKernel{I}() where I
        @assert I ∈ (-1, 0, 1, 2, 3) "Invalid parameter i=$(I). Should be -1, 0, 1, 2 or 3."
        if I == -1
            return WhiteKernel()
        end
        return new{I}()
    end
end

function WienerKernel(;i::Integer=0)
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
    return 1 / 20 * minXY^5 + 1 / 12 * minXY^3 * euclidean(x, y) *
        ( X + Y - 1 / 2 * minXY )
end

function (::WienerKernel{3})(x, y)
    X = sqrt(sum(abs2, x))
    Y = sqrt(sum(abs2, y))
    minXY = min(X, Y)
    return 1 / 252 * minXY^7 + 1 / 720 * minXY^4 * euclidean(x, y) *
        ( 5 * max(X, Y)^2 + 2 * X * Y + 3 * minXY^2 )
end

Base.show(io::IO, κ::WienerKernel{I}) where I = print(io, "Wiener Kernel ", I, "-times integrated")
