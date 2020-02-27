struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

kernel(κ) = κ.kernel

kappa(κ::TransformedKernel, x) = kappa(κ.kernel, x)

metric(κ::TransformedKernel) = metric(κ.kernel)

params(κ::TransformedKernel) = (params(κ.transform),params(κ.kernel))

Base.show(io::IO,κ::TransformedKernel) = print(io,"$(κ.kernel)\n\t- $(κ.transform)")
