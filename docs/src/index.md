# KernelFunctions.jl

Model agnostic kernel functions compatible with automatic differentiation

**KernelFunctions.jl** is a general purpose kernel package.
It aims at providing a flexible framework for creating kernels and manipulating them.
The main goals of this package compared to its predecessors/concurrents in [MLKernels.jl](https://github.com/trthatcher/MLKernels.jl), [Stheno.jl](https://github.com/willtebbutt/Stheno.jl), [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) and [AugmentedGaussianProcesses.jl](https://github.com/theogf/AugmentedGaussianProcesses.jl) are:
- **Automatic Differentation** compatibility: all kernel functions should be differentiable via packages like [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) or [Zygote.jl](https://github.com/FluxML/Zygote.jl)
- **Flexibility**: operations between kernels should be fluid and easy without breaking.
- **Plug-and-play**: including the kernels before/after other steps should be straightforward.

The methodology of how kernels are computed is quite simple and is done in three phases :
- A `Transform` object is applied sample-wise on every sample
- The pairwise matrix is computed using [Distances.jl](https://github.com/JuliaStats/Distances.jl) by using a `Metric` proper to each kernel
- The `Kernel` function is applied element-wise on the pairwise matrix

For a quick introduction on how to use it go to [User guide](@ref)
