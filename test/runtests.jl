using Test
using KernelFunctions
using Distances
using Random

@testset "KernelFunctions" begin
include("test_kernelmatrix.jl")
include("test_approximations.jl")
include("test_constructors.jl")
# include("test_AD.jl")
include("test_transform.jl")
include("test_distances.jl")
include("test_kernels.jl")
include("test_generic.jl")
include("test_adjoints.jl")
    #include("types.jl")
end
