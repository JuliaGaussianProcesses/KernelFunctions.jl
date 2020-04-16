"""
    ScaledKernel(k::Kernel, σ²::Real)

Return a kernel premultiplied by the variance `σ²` : `σ² k(x,x')`
"""
struct ScaledKernel{Tk<:Kernel, Tσ²<:Real} <: Kernel
    kernel::Tk
    σ²::Vector{Tσ²}
end

function ScaledKernel(kernel::Tk, σ²::Tσ²=1.0) where {Tk<:Kernel,Tσ²<:Real}
    @check_args(ScaledKernel, σ², σ² > zero(Tσ²), "σ² > 0")
    return ScaledKernel{Tk, Tσ²}(kernel, [σ²])
end

kappa(k::ScaledKernel, x) = first(k.σ²) * kappa(k.kernel, x)

kappa(k::ScaledKernel, x, y) = first(k.σ²) * kappa(k.kernel, x, y)

metric(k::ScaledKernel) = metric(k.kernel)

Base.:*(w::Real, k::Kernel) = ScaledKernel(k, w)

Base.show(io::IO, κ::ScaledKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::ScaledKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    print(io,"\n" * ("\t"^(shift+1)) * "- σ² = $(first(κ.σ²))")
end
