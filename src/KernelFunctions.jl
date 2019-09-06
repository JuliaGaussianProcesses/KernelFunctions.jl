module KernelFunctions

export kernelmatrix, kernelmatrix!, kappa
export Kernel, SquaredExponentialKernel, MaternKernel, Matern3_2Kernel, Matern5_2Kernel

export Transform, ScaleTransform

using Distances, LinearAlgebra
using Zygote: @adjoint
using SpecialFunctions: lgamma, besselk
using StatsFuns: logtwo

const defaultobs = 2
abstract type Kernel{T,Tr} end

include("zygote_rules.jl")
include("utils.jl")
include("common.jl")
include("transform/transform.jl")
include("kernelmatrix.jl")

kernels = ["squaredexponential","matern"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

end
