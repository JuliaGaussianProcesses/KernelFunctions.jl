struct LinearSplineKernel{Tc<:Real} <: Kernel
    c::Tc
    function LinearSplineKernel(c::Tc) where {Tc<:Real}
        @check_args(LinearSplineKernel, c, zero(c) ≤ c, "c ≥ 0")
        return new{Tc}(c)
    end
end

(k::LinearSplineKernel)(x::Real, y::Real) = 1 + k.c - (k.c / 10) * abs(x - y)

# Specialised implementations required for Zygote performance.

function kernelmatrix(k::LinearSplineKernel, x::AbstractVector{<:Real})
    c = k.c
    c10 = c / 10
    return map(d -> 1 + c - c10 * d, pairwise(Euclidean(), x))
end

function kernelmatrix(
    k::LinearSplineKernel, x::AbstractVector{<:Real}, y::AbstractVector{<:Real}
)
    c = k.c
    c10 = c / 10
    return map(d -> 1 + c - c10 * d, pairwise(Euclidean(), x, y))
end

# Necessary for performance on 1.6.
function kernelmatrix_diag(k::LinearSplineKernel, x::AbstractVector{<:Real})
    return fill(1 + k.c, length(x))
end
