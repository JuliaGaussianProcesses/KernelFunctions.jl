# KernelFunctions.jl

**KernelFunctions.jl** is a general purpose [kernel](https://en.wikipedia.org/wiki/Positive-definite_kernel) package.
It aims at providing a flexible framework for creating kernels and manipulating them.
the main goals of this package are:
- **Flexibility**: operations between kernels should be fluid and easy without breaking.
- **Plug-and-play**: including the kernels before/after other steps should be straightforward.
- **Automatic Differentation** compatibility: all kernel functions which _ought_ to be differentiable using AD packages like [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) or [Zygote.jl](https://github.com/FluxML/Zygote.jl) _should_ be.

This package builds on of lots of excellent existing work in packages such as [MLKernels.jl](https://github.com/trthatcher/MLKernels.jl), [Stheno.jl](https://github.com/willtebbutt/Stheno.jl), [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl), and [AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl).

See the [User guide](@ref) for a brief introduction.
