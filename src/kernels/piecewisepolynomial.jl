"""
    PiecewisePolynomialKernel{V}(maha::AbstractMatrix)

Piecewise Polynomial covariance function with compact support, V = 0,1,2,3.
The kernel functions are 2v times continuously differentiable and the corresponding
processes are hence v times  mean-square differentiable. The kernel function is:
```math
    κ(x,y) = max(1-r,0)^(j+V) * f(r,j) with j = floor(D/2)+V+1
```
where r is the Mahalanobis distance mahalanobis(x,y) with `maha` as the metric.

"""
struct PiecewisePolynomialKernel{V, A<:AbstractMatrix{<:Real}} <: BaseKernel
    maha::A
    function PiecewisePolynomialKernel{V}(maha::AbstractMatrix{<:Real}) where V
        V in (0, 1, 2, 3) || error("Invalid paramter v=$(V). Should be 0, 1, 2 or 3.")
        LinearAlgebra.checksquare(maha)
        new{V,typeof(maha)}(maha)
    end
end

function _f(κ::PiecewisePolynomialKernel{V}, r, j) where V
    if V==0
        return 1
    elseif V==1
        return 1 + (j+1)*r
    elseif V==2
        return 1 + (j+2)*r +   (j^2+ 4*j+ 3)/ 3*r.^2
    elseif V==3
        return 1 + (j+3)*r +   (6*j^2+36*j+45)/15*r.^2 + (j^3 + 9*j^2 + 23*j + 15)/15 * r.^3
    else
        error("Invalid paramter v=$(V). Should be 0,1,2 or 3.")
    end
end

function kappa(κ::PiecewisePolynomialKernel{V}, r::T) where {T<:Real,V} 
    j = div(size(r, 2), 2) + V + 1
    return max(1-r,0)^(j + V) * _f(κ,r,j)
end
metric(κ::PiecewisePolynomialKernel) = Mahalanobis(κ.maha)

Base.show(io::IO, κ::PiecewisePolynomialKernel{V}) where V = print(io, "Piecewise Polynomial Kernel (v = $(V), size(maha) = $(size(κ.maha))")
