using Test
using Distances, LinearAlgebra
using KernelFunctions

A = rand(10,5)
B = rand(20,5)
@testset "Distance" begin
    @testset "Dot Product" begin
        d = KernelFunctions.DotProduct()
        @test diag(pairwise(d,A,dims=2)) == dot.(eachcol(A),eachcol(A))
        @test_throws DimensionMismatch d(rand(3),rand(4))
        @test d(3.0,2.0) == 6.0
    end
    @testset "Delta" begin
        d = KernelFunctions.Delta()
        @test pairwise(d,A,dims=1) == Matrix(I,size(A,1),size(A,1))
        @test pairwise(d,A,B,dims=1) == zeros(size(A,1),size(B,1))
        @test d(1,2) == 0
        @test d(1,1) == 1
        @test_throws DimensionMismatch d(rand(3),rand(4))
    end
end
