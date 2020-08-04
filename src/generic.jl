## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k,nothing)
Base.iterate(k::Kernel, ::Any) = nothing

printshifted(io::IO, o, shift::Int) = print(io, o)

### Syntactic sugar for creating matrices and using kernel functions
function concretetypes(k, ktypes::Vector)
    isempty(subtypes(k)) ? push!(ktypes, k) : concretetypes.(subtypes(k), Ref(ktypes))
    return ktypes
end

# Fallback implementation of evaluate for `SimpleKernel`s.
(k::SimpleKernel)(x, y) = kappa(k, evaluate(metric(k), x, y))
