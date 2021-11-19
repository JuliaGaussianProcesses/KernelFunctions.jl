## Allows to iterate over kernels
Base.length(::Kernel) = 1
Base.iterate(k::Kernel) = (k, nothing)
Base.iterate(k::Kernel, ::Any) = nothing

function print_nested(io::IO, x)
    toplevel = get(io, :KERNELFUNCTIONS_TOP, true)::Bool
    if toplevel
        recur_io = IOContext(io, :KERNELFUNCTIONS_TOP => false)
        print_toplevel(recur_io, x)
    else
        print(io, "(")
        print_toplevel(io, x)
        print(io, ")")
    end
    return nothing
end

# Fallback implementation of evaluate for `SimpleKernel`s.
(k::SimpleKernel)(x, y) = kappa(k, metric(k)(x, y))
