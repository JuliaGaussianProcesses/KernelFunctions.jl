@testset "maha" begin
    rng = MersenneTwister(123456)
    x = 2 * rand(rng)
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    P = rand(rng, 3, 3)
    k = MahalanobisKernel(P=P)

    @test kappa(k, x) == exp(-x)
    @test k(v1, v2) ≈ exp(-sqmahalanobis(v1, v2, P))
    @test kappa(ExponentialKernel(), x) == kappa(k, x)
    @test repr(k) == "Mahalanobis Kernel (size(P) = $(size(P)))"
    # test_ADs(P -> MahalanobisKernel(P=P), P)
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"
end
