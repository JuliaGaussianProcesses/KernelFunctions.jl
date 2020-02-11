struct TransformedKernel{Tk<:Kernel,Tr<:Transform} <: Kernel
    kernel::Tk
    transform::Tr
end

kernel(κ) = κ.kernel

transform(κ::Kernel,t::Transform) = TransformedKernel(κ,t)

kappa(κ::TransformedKernel, x) = kappa(κ.kernel, x)

metric(κ::TransformedKernel) = metric(κ.kernel)

params(κ::TransformedKernel) = (params(κ.transform),params(κ.kernel))
opt_params(κ::TransformedKernel) = (opt_params(κ.transform),opt_params(κ.kernel))

Base.show(io::IO,κ::TransformedKernel) = print(io,"$(κ.kernel) with $(κ.transform)")
