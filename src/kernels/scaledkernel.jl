struct ScaledKernel{Tk<:Kernel,Tσ<:Real} <: Kernel
    kernel::Tk
    σ::Vector{Tσ}
end

function ScaledKernel(kernel::Tk,σ::Tσ=1.0) where {Tk<:Kernel,Tσ<:Real}
    @check_args(ScaledKernel, σ, σ > zero(Tσ), "σ > 0")
    ScaledKernel{Tk,Tσ}(kernel,[σ])
end

kappa(k::ScaledKernel, x) = first(k.σ)*kappa(k.kernel, x)

metric(k::ScaledKernel) = metric(k.kernel)

params(k::ScaledKernel) = (k.σ,params(k.kernel))
opt_params(k::ScaledKernel) = (k.σ,opt_params(k.kernel))

Base.:*(w::Real,k::Kernel) = ScaledKernel(k,w)

Base.show(io::IO,κ::ScaledKernel) = printshifted(io,κ,0)

function printshifted(io::IO,κ::ScaledKernel,shift::Int)
    printshifted(io,κ.kernel,shift)
    print(io,"\n"*("\t"^(shift+1))*"- σ = $(first(κ.σ))")
end
