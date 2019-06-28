module KernelFunctions

export kernelmatrix, kernelmatrix!, kappa
export Kernel, SquaredExponentialKernel

export Transform, ScaleTransform

using Distances, LinearAlgebra

const defaultobs = 2
abstract type Kernel{T,Tr} end

include("utils.jl")
include("common.jl")
include("transform/transform.jl")
include("kernelmatrix.jl")

kernels = ["squaredexponential"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

end
