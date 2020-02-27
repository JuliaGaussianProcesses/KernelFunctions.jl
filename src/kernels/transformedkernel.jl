struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

kernel(κ) = κ.kernel

kappa(κ::TransformedKernel, x) = kappa(κ.kernel, x)

metric(κ::TransformedKernel) = metric(κ.kernel)

params(κ::TransformedKernel) = (params(κ.transform),params(κ.kernel))

Base.show(io::IO,κ::TransformedKernel) = printshifted(io,κ,0)

function printshifted(io::IO,κ::TransformedKernel,shift::Int)
    printshifted(io,κ.kernel,shift)
    print(io,"\n"*("\t"^(shift+1))*"- $(κ.transform)")
end
