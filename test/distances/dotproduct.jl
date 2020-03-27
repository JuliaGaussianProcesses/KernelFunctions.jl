@testset "Dot Product" begin
    A = rand(10,5)
    B = rand(20,5)
    d = KernelFunctions.DotProduct()
    @test diag(pairwise(d,A,dims=2)) == [dot(A[:,i],A[:,i]) for i in 1:size(A,2)]
    @test_throws DimensionMismatch d(rand(3),rand(4))
    @test d(3.0,2.0) == 6.0
end
