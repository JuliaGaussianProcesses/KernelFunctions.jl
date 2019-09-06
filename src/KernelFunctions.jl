module KernelFunctions

export kernelmatrix, kernelmatrix!, kerneldiagmatrix, kerneldiagmatrix!, kappa
export Kernel, SquaredExponentialKernel, MaternKernel, Matern32Kernel, Matern52Kernel

export Transform, ScaleTransform

using Distances, LinearAlgebra
using Zygote: @adjoint
using SpecialFunctions: lgamma, besselk
using StatsFuns: logtwo

const defaultobs = 2
abstract type Kernel{T,Tr} end

include("zygote_rules.jl")
include("utils.jl")
include("transform/transform.jl")
include("kernelmatrix.jl")

kernels = ["squaredexponential","matern"]
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

include("generic.jl")


end
