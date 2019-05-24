module KernelFunctions

using Distances, LinearAlgebra

const defaultobs = 2
abstract type Kernel{T} where {T<:Real} end

include("kernelmatrix.jl")
include("kernels/common.jl")

kernels = ("squaredexponential")
for k in kernels
    include(joinpath("kernels",k*".jl"))
end

end
