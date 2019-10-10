using Test
using KernelFunctions
using Distances
using FiniteDifferences
using Random
using Zygote

# Helpful functionality for writing tests.
include("test_util.jl")

@testset "KernelFunctions" begin
include("test_kernelmatrix.jl")
include("test_constructors.jl")
# include("test_AD.jl")
include("test_transform.jl")
    #include("types.jl")
end
