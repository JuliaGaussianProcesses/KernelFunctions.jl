@testset "pairwise" begin
    rng = MersenneTwister(123456)
    d = SqEuclidean()

    x = [randn(rng, 3) for _ in 1:4]
    y = [randn(rng, 3) for _ in 1:5]
    X = hcat(x...)
    Y = hcat(y...)
    K = zeros(4, 5)

    @test KernelFunctions.pairwise(d, x, y) ≈ pairwise(d, X, Y, dims=2)
    @test KernelFunctions.pairwise(d, x) ≈ pairwise(d, X, dims=2)
    KernelFunctions.pairwise!(K, d, x, y)
    @test K ≈ pairwise(d, X, Y, dims=2)

    x = randn(rng, 10)
    X = reshape(x, :, 1)
    y = randn(rng, 11)
    Y = reshape(y, :, 1)
    K = zeros(10, 11)
    @test KernelFunctions.pairwise(d, x, y) ≈ pairwise(d, X, Y; dims=1)
    @test KernelFunctions.pairwise(d, x) ≈ pairwise(d, X; dims=1)
    KernelFunctions.pairwise!(K, d, x, y)
    @test K ≈ pairwise(d, X, Y, dims=1)


end
