struct LinearSplineKernel{Tc<:Real} <: Kernel
    c::Tc
    function LinearSplineKernel(c::Tc) where {Tc<:Real}
        @check_args(LinearSplineKernel, c, zero(c) ≤ c, "c ≥ 0")
        return new{Tc}(c)
    end
end

(k::LinearSplineKernel)(x::Real, y::Real) = 1 + k.c - (k.c / 10) * abs(x - y)
