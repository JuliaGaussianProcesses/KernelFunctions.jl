"""
    WienerKernel{i}()

i-times integrated Wiener process kernel function.

- For i=-1, this is just the white noise covariance, see [`WhiteKernel`](@ref).
- For i= 0, this is the Wiener process covariance,
- For i= 1, this is the integrated Wiener process covariance (velocity),
- For i= 2, this is the twice-integrated Wiener process covariance (accel.),
- For i= 3, this is the thrice-integrated Wiener process covariance,

where ``κᵢ`` is given by

```math
    κ₋₁(x, y) =  δ(x, y)
    κ₀(x, y)  =  min(x, y)
```
and for ``i >= 1``,
```math
    κᵢ(x, y) = 1 / aᵢ * min(x, y)^(2i + 1) + bᵢ * min(x, y)^(i + 1) * |x - y| * rᵢ(x, y),
```
    with the coefficients ``aᵢ``, ``bᵢ`` and the residual ``rᵢ(x, y)`` defined as follows:
```math
    a₁ = 3, b₁ = 1/2, r₁(x, y) = 1,
    a₂ = 20, b₂ = 1/12, r₂(x, y) = x + y - min(x, y) / 2,
    a₃ = 252, b₃ = 1/720, r₃(x, y) = 5 * max(x, y)² + 2 * x * z + 3 * min(x, y)²

```

# References:
See the paper *Probabilistic ODE Solvers with Runge-Kutta Means* by Schober, Duvenaud and
Hennig, NIPS, 2014, for more details.

"""
struct WienerKernel{I} <: Kernel
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
    X_2 = sum(abs2, x)
    Y_2 = sum(abs2, y)
    return sqrt(min(X_2, Y_2))
end

function (::WienerKernel{1})(x, y)
    X_2 = sum(abs2, x)
    Y_2 = sum(abs2, y)
    minX2Y2 = min(X_2, Y_2)
    return 1 / 3 * minX2Y2^(3/2) + 1 / 2 * minX2Y2 * euclidean(x, y)
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

function kernelmatrix(::WienerKernel{I}, x::ColVecs, y::ColVecs) where I
    validate_inputs(x, y)
    X = sqrt.(sum(x.X .* x.X; dims=1))
    Y = sqrt.(sum(y.X .* y.X; dims=1))
    minXY = min.(permutedims(X), Y)
    if I == 0
        return minXY
    elseif I == 1
        return (1 / 3) .* minXY.^3 .+ (1 / 2) .* minXY.^2 .* pairwise(Euclidean(), x, y)
    elseif I == 2
        return (1 / 20) .* minXY.^5 .+ (1 / 12) .* minXY.^3 .* pairwise(Euclidean(), x, y) .*
            ( X + Y .- (1 / 2) .* minXY )
    elseif I == 3
        return (1 / 252) .* minXY.^7 .+ (1 / 720) .* minXY.^4 .* pairwise(Euclidean(), x, y) .*
            ( 5 .* max.(permutedims(X), Y).^2 .+ 2 .* X .* Y .+ 3 .* minXY.^2 )
    end
    return error("Invalid I=$I")
end

function kernelmatrix(::WienerKernel{I}, x::RowVecs, y::RowVecs) where I
    validate_inputs(x, y)
    X = sqrt.(sum(x.X .* x.X; dims=2))
    Y = sqrt.(sum(y.X .* y.X; dims=2))
    minXY = min.(permutedims(X), Y)
    if I == 0
        return minXY
    elseif I == 1
        return (1 / 3) .* minXY.^3 .+ (1 / 2) .* minXY.^2 .* pairwise(Euclidean(), x, y)
    elseif I == 2
        return (1 / 20) .* minXY.^5 .+ (1 / 12) .* minXY.^3 .* pairwise(Euclidean(), x, y) .*
            ( X .+ Y .- (1 / 2) .* minXY )
    elseif I == 3
        return (1 / 252) .* minXY.^7 .+ (1 / 720) .* minXY.^4 .* pairwise(Euclidean(), x, y) .*
            ( 5 .* max.(permutedims(X), Y).^2 .+ 2 .* X .* Y .+ 3 .* minXY.^2 )
    end
    return error("Invalid I=$I")
end

Base.show(io::IO, ::WienerKernel{I}) where I = print(io, I, "-times integrated Wiener kernel")
