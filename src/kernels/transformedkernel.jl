struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

@inline transform(k::TransformedKernel) = k.transform

@inline kappa(k::TransformedKernel, x) = kappa(k.kernel, x)

@inline metric(k::TransformedKernel) = metric(k.kernel)

params(k::TransformedKernel) = (params(k.transform),params(k.kernel))
opt_params(k::TransformedKernel) = (opt_params(k.transform),opt_params(k.kernel))
