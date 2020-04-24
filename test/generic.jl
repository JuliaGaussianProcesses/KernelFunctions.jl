@testset "generic" begin
    k = SqExponentialKernel()
    @test length(k) == 1
    @test iterate(k) == (k,nothing)
    @test iterate(k,1) == nothing

    rng = MersenneTwister(123456)
    x = randn(rng, 10)
    X = reshape(x, :, 1)
    y = randn(rng, 11)
    Y = reshape(y, :, 1)
    @test pairwise(SqEuclidean(), x, y) ≈ pairwise(SqEuclidean(), X, Y; dims=1)
    @test pairwise(SqEuclidean(), x) ≈ pairwise(SqEuclidean(), X; dims=1)
end
