using Distances, LinearAlgebra
using Test
using KernelFunctions

dims = [10,5]
X = rand(dims...)
k = SqExponentialKernel()
@testset "Kernel Matrix Approximations" begin
    @testset "Nystrom" begin
        for obsdim in [1, 2]
            @test kernelmatrix(k, X; obsdim=obsdim) ≈ kernelmatrix(nystrom(k, X, 1.0; obsdim=obsdim))
            @test kernelmatrix(k, X; obsdim=obsdim) ≈ kernelmatrix(nystrom(k, X, collect(1:dims[obsdim]); obsdim=obsdim))
        end
    end
end
