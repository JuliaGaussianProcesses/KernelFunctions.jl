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

@functor ScaledKernel

(k::ScaledKernel)(x, y) = first(k.σ²) * k.kernel(x, y)

function kernelmatrix(κ::ScaledKernel, x::AbstractVector, y::AbstractVector)
    return κ.σ² .* kernelmatrix(κ.kernel, x, y)
end

function kernelmatrix(κ::ScaledKernel, x::AbstractVector)
    return κ.σ² .* kernelmatrix(κ.kernel, x)
end

function kerneldiagmatrix(κ::ScaledKernel, x::AbstractVector)
    return κ.σ² .* kerneldiagmatrix(κ.kernel, x)
end

function kernelmatrix!(
    K::AbstractMatrix, κ::ScaledKernel, x::AbstractVector, y::AbstractVector,
)
    kernelmatrix!(K, κ, x, y)
    K .*= κ.σ² 
    return K
end

function kernelmatrix!(K::AbstractMatrix, κ::ScaledKernel, x::AbstractVector)
    kernelmatrix!(K, κ, x)
    K .*= κ.σ²
    return K
end

function kerneldiagmatrix!(K::AbstractVector, κ::ScaledKernel, x::AbstractVector)
    kerneldiagmatrix!(K, κ, x)
    K .*= κ.σ²
    return K
end

Base.:*(w::Real, k::Kernel) = ScaledKernel(k, w)

Base.show(io::IO, κ::ScaledKernel) = printshifted(io, κ, 0)

function printshifted(io::IO, κ::ScaledKernel, shift::Int)
    printshifted(io, κ.kernel, shift)
    print(io,"\n" * ("\t"^(shift+1)) * "- σ² = $(first(κ.σ²))")
end
