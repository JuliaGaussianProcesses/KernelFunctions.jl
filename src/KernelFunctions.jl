module KernelFunctions

export kernelmatrix, kernelmatrix!, kappa
export Kernel, SquaredExponentialKernel

using Distances, LinearAlgebra

const defaultobs = 2
abstract type Kernel{T<:Real} end

include("utils.jl")
include("common.jl")
include("kernelmatrix.jl")

kernels = ["squaredexponential"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

end
