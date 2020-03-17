using Test
using Distances, LinearAlgebra
using KernelFunctions

A = rand(10,5)
B = rand(20,5)
@testset "Distance" begin
    @testset "Dot Product" begin
        d = KernelFunctions.DotProduct()
        @test diag(pairwise(d,A,dims=2)) == [dot(A[:,i],A[:,i]) for i in 1:size(A,2)]
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
    @testset "Sinus" begin
        d = KernelFunctions.Sinus(ones(5))
        @test d(1,1) == 0
        @test d(0,1) ≈ 0 atol=1e-16
        @test d(0,0.5) ≈ 1
        @test_throws DimensionMismatch d(rand(5),rand(4))
        @test_throw DimensionMismatch d(rand(3),rand(3))
    end
end
