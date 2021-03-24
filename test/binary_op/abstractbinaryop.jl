@testset "abstractbinaryop" begin
    using KernelFunctions: AbstractBinaryOp
    rng = MersenneTwister(123456)
    d = SqEuclidean()
    Ns = (4, 5)
    D = 3
    x = [randn(rng, D) for _ in 1:Ns[1]]
    y = [randn(rng, D) for _ in 1:Ns[2]]
    X = hcat(x...)
    Y = hcat(y...)
    K = zeros(Ns)

    struct Max <: AbstractBinaryOp end

    (d::Max)(a, b) =  Distances.evaluate(d, a, b)
    Distances.evaluate(::Max, a, b) = maximum(abs, a - b)
    @test pairwise(d, x, y) ≈ pairwise(d, X, Y; dims=2)
    @test pairwise(d, x) ≈ pairwise(d, X; dims=2)
    pairwise!(K, d, x, y)
    @test K ≈ pairwise(d, X, Y; dims=2)
    K = zeros(Ns[1], Ns[1])
    pairwise!(K, d, x)
    @test K ≈ pairwise(d, X; dims=2)

    x = randn(rng, 10)
    X = reshape(x, :, 1)
    y = randn(rng, 11)
    Y = reshape(y, :, 1)
    K = zeros(10, 11)
    @test pairwise(d, x, y) ≈ pairwise(d, X, Y; dims=1)
    @test pairwise(d, x) ≈ pairwise(d, X; dims=1)
    pairwise!(K, d, x, y)
    @test K ≈ pairwise(d, X, Y; dims=1)
end
