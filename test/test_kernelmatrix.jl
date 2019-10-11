using Distances, LinearAlgebra
using Test
using KernelFunctions

dims = [10,5]

A = rand(dims...)
B = rand(dims...)
C = rand(8,9)
K = [zeros(dims[1],dims[1]),zeros(dims[2],dims[2])]
Kdiag = [zeros(dims[1]),zeros(dims[2])]
k = SqExponentialKernel()
@testset "Kernel Matrix Operations" begin
    @testset "Inplace Kernel Matrix" begin
        for obsdim in [1,2]
            @test kernelmatrix!(K[obsdim],k,A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
            @test kernelmatrix!(K[obsdim],k,A,obsdim=obsdim) == kernelmatrix(k,A,obsdim=obsdim)
            @test kerneldiagmatrix!(Kdiag[obsdim],k,A,obsdim=obsdim) == kerneldiagmatrix(k,A,obsdim=obsdim)
            @test_throws DimensionMismatch kernelmatrix!(K[obsdim],k,A,C,obsdim=obsdim)
            @test_throws DimensionMismatch kernelmatrix!(K[obsdim],k,C,obsdim=obsdim)
            @test_throws DimensionMismatch kerneldiagmatrix!(Kdiag[obsdim],k,C,obsdim=obsdim)
        end
    end
    @testset "Kernel matrix" begin
        for obsdim in [1,2]
            @test kernelmatrix(k,A,B,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,B,dims=obsdim))
            @test kernelmatrix(k,A,obsdim=obsdim) == kappa.([k],pairwise(KernelFunctions.metric(k),A,dims=obsdim))
            @test k(A,B,obsdim=obsdim) == kernelmatrix(k,A,B,obsdim=obsdim)
            @test k(A,obsdim=obsdim) == kernelmatrix(k,A,obsdim=obsdim)
            @test kernel(k,1.0,2.0) == kernel(k,[1.0],[2.0])
            @test_throws DimensionMismatch kernelmatrix(k,A,C,obsdim=obsdim)
        end
    end
end
