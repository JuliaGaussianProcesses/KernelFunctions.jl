# KernelFunctions.jl

**KernelFunctions.jl** is a general purpose [kernel](https://en.wikipedia.org/wiki/Positive-definite_kernel) package.
It provides a flexible framework for creating kernel functions and manipulating them, and an extensive collection of implementations.
The main goals of this package are:
- **Flexibility**: operations between kernels should be fluid and easy without breaking, with a user-friendly API.
- **Plug-and-play**: being model-agnostic; including the kernels before/after other steps should be straightforward. To interoperate well with generic packages for handling parameters like [ParameterHandling.jl](https://github.com/invenia/ParameterHandling.jl/) and FluxML's [Functors.jl](https://github.com/FluxML/Functors.jl/).
- **Automatic Differentiation compatibility**: all kernel functions which _ought_ to be differentiable using AD packages like [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) or [Zygote.jl](https://github.com/FluxML/Zygote.jl) _should_ be.

This package replaces the now-defunct [MLKernels.jl](https://github.com/trthatcher/MLKernels.jl). It incorporates lots of excellent existing work from packages such as [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl), and is used in downstream packages such as [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl), [ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl), [Stheno.jl](https://github.com/willtebbutt/Stheno.jl), and [AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl).

See the [User guide](@ref) for a brief introduction.
