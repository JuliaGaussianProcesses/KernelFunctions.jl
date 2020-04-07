"""
    WienerKernel{i}()

i-times integrated Wiener process kernel function given by
```julia
    κ(x,y) =  kᵢ(x,y)
```

For i=-1, this is just the white noise covariance, see WhiteKernel.\\
For i= 0, this is the Wiener process covariance,\\
for i= 1, this is the integrated Wiener process covariance (velocity),\\
for i= 2, this is the twice-integrated Wiener process covariance (accel.),\\
for i= 3, this is the thrice-integrated Wiener process covariance. where `kᵢ` is given by\\

```julia
    k₋₁(x,y) =  δ(x,y)
    i >= 0, kᵢ(x,y) = 1/ai * min(x,y)^(2i + 1) + bi * min(x,y)^(i+1) * |x-y| * ri(x,y),
    with the coefficients ai, bi and the residual ri(x,y) defined as follows:
        i = 0, ai =   1, bi = 0
        i = 1, ai =   3, bi = 1/  2, ri(x,y) = 1
        i = 2, ai =  20, bi = 1/ 12, ri(x,y) = x + y - 1/2 * min(x,y)
        i = 3, ai = 252, bi = 1/720, ri(x,y) = 5 * max(x,y)² + 2xz + 3min(x,y)²
```

**References:**\\
See the paper *Probabilistic ODE Solvers with Runge-Kutta Means* by Schober, Duvenaud and Hennig, NIPS, 2014, for more details.

"""
struct WienerKernel{I} <: BaseKernel
    function WienerKernel{I}() where I
        I in (-1, 0, 1, 2, 3) || error("Invalid paramter i=$(I). Should be -1, 0, 1, 2 or 3.")
        if I==-1
            return WhiteKernel()
        end
        new{I}()
    end
end

function WienerKernel(;i=0)
    return WienerKernel{i}()
end

function _wiener(κ::WienerKernel{I},x,y) where I
    X = sqrt(sum(abs2.(x)))
    Y = sqrt(sum(abs2.(y)))
    minXY = min(X,Y)
    if I==0
        return         minXY^(2I + 1)
    elseif I==1
        return 1/3   * minXY^(2I + 1) + 1/2   * minXY^(I+1) * euclidean(x,y)
    elseif I==2
        return 1/20  * minXY^(2I + 1) + 1/12  * minXY^(I+1) * euclidean(x,y) * (X + Y - 1/2 * minXY)
    elseif I==3
        return 1/252 * minXY^(2I + 1) + 1/720 * minXY^(I+1) * euclidean(x,y) * (5*max(X,Y)^2 + 2*X*Y + 3 * minXY^2)
    else
        error("Invalid I")
    end
end

function kappa(κ::WienerKernel, x,y)
    return _wiener(κ, x, y)
end

function _kernel(
    κ::WienerKernel,
    x::AbstractVector,
    y::AbstractVector;
    obsdim::Int = defaultobs
)
    @assert length(x) == length(y) "x and y don't have the same dimension!"
    kappa(κ,x,y)
end

function kernelmatrix(
    κ::WienerKernel,
    X::AbstractMatrix;
    obsdim::Int = defaultobs
)
    return map(r->_piecewisepolynomial(κ,r,j),pairwise(metric(κ),X,dims=obsdim))
end


Base.show(io::IO, κ::WienerKernel{I}) where I = print(io, "Wiener Kernel $(I)-times integrated")
