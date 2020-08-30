@testset "maha" begin
    rng = MersenneTwister(123456)
    x = 2 * rand(rng)
    D_in = 3
    v1 = rand(rng, D_in)
    v2 = rand(rng, D_in)

    P_ = randn(3, 3)
    P = collect(Symmetric(P_ * P_' + I))
    k = MahalanobisKernel(P=P)

    @test kappa(k, x) == exp(-x)
    @test k(v1, v2) ≈ exp(-sqmahalanobis(v1, v2, P))
    @test kappa(ExponentialKernel(), x) == kappa(k, x)
    @test repr(k) == "Mahalanobis Kernel (size(P) = $(size(P)))"
    # test_ADs(P -> MahalanobisKernel(P=P), P)
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"

    # Standardised tests.
    @testset "ColVecs" begin
        x0 = ColVecs(randn(D_in, 3))
        x1 = ColVecs(randn(D_in, 3))
        x2 = ColVecs(randn(D_in, 2))
        TestUtils.test_interface(k, Float64)
    end
    @testset "RowVecs" begin
        x0 = ColVecs(randn(3, D_in))
        x1 = ColVecs(randn(3, D_in))
        x2 = ColVecs(randn(2, D_in))
        TestUtils.test_interface(k, Float64)
    end
    test_params(k, (P,))
end
