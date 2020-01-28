struct ScaledKernel{Tk<:Kernel,Tσ<:Real} <: Kernel
    kernel::Tk
    σ::Vector{Tσ}
end

function ScaledKernel(kernel::Tk,σ::Tσ=1.0) where {Tk<:Kernel,Tσ<:Real}
    @check_args(ScaledKernel, σ, σ > zero(Tσ), "σ > 0")
    ScaledKernel{Tk,Tσ}(kernel,[σ])
end

@inline transform(k::ScaledKernel) = transform(k.kernel)

@inline kappa(k::ScaledKernel, x) = first(k.σ)*kappa(k.kernel, x)

@inline metric(k::ScaledKernel) = metric(k.kernel)

params(k::ScaledKernel) = (k.σ,params(k.kernel))
opt_params(k::ScaledKernel) = (k.σ,opt_params(k.kernel))

Base.:*(w::Real,k::Kernel) = ScaledKernel(k,w)
