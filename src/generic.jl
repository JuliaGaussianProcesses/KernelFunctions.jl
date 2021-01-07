## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k, nothing)
Base.iterate(k::Kernel, ::Any) = nothing

printshifted(io::IO, o, shift::Int) = print(io, o)

# Fallback implementation of evaluate for `SimpleKernel`s.
(k::SimpleKernel)(x, y) = kappa(k, evaluate(metric(k), x, y))
