using Test
using KernelFunctions
using Distances
using FiniteDifferences
using Random
using Zygote

# Helpful functionality for writing tests.
include("test_util.jl")

@testset "KernelFunctions" begin
    include("zygote_rules.jl")
    include("kernelmatrix.jl")
    include("constructors.jl")
    include("testAD.jl")
    #include("types.jl")
end
