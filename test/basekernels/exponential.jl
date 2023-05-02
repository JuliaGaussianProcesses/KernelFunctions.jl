@testset "exponential" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @testset "SqExponentialKernel" begin
        k = SqExponentialKernel()
        @test kappa(k, x) ≈ exp(-x / 2)
        @test k(v1, v2) ≈ exp(-norm(v1 - v2)^2 / 2)
        @test kappa(SqExponentialKernel(), x) == kappa(k, x)
        @test metric(SqExponentialKernel()) == SqEuclidean()
        @test RBFKernel == SqExponentialKernel
        @test GaussianKernel == SqExponentialKernel
        @test SEKernel == SqExponentialKernel
        @test repr(k) == "Squared Exponential Kernel (metric = Euclidean(0.0))"
        @test KernelFunctions.iskroncompatible(k) == true

        k2 = SqExponentialKernel(; metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k)
        test_ADs(SEKernel)
        test_interface_ad_perf(_ -> SEKernel(), nothing, StableRNG(123456))
    end
    @testset "ExponentialKernel" begin
        k = ExponentialKernel()
        @test kappa(k, x) ≈ exp(-x)
        @test k(v1, v2) ≈ exp(-norm(v1 - v2))
        @test kappa(ExponentialKernel(), x) == kappa(k, x)
        @test metric(ExponentialKernel()) == Euclidean()
        @test repr(k) == "Exponential Kernel (metric = Euclidean(0.0))"
        @test LaplacianKernel == ExponentialKernel
        @test KernelFunctions.iskroncompatible(k) == true

        k2 = ExponentialKernel(; metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k)
        test_ADs(ExponentialKernel)
        test_interface_ad_perf(_ -> ExponentialKernel(), nothing, StableRNG(123456))
    end
    @testset "GammaExponentialKernel" begin
        γ = 1.0
        k = GammaExponentialKernel(; γ=γ)
        @test k(v1, v2) ≈ exp(-norm(v1 - v2)^γ)
        @test kappa(GammaExponentialKernel(), x) == kappa(k, x)
        @test GammaExponentialKernel(; gamma=γ).γ == [γ]
        @test metric(GammaExponentialKernel()) == Euclidean()
        @test metric(GammaExponentialKernel(; γ=2.0)) == Euclidean()
        @test repr(k) == "Gamma Exponential Kernel (γ = $(γ), metric = Euclidean(0.0))"
        @test KernelFunctions.iskroncompatible(k) == true

        k2 = GammaExponentialKernel(; γ=γ, metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        test_ADs(γ -> GammaExponentialKernel(; gamma=only(γ)), [1 + 0.5 * rand()])
        test_params(k, ([γ],))
        TestUtils.test_interface(GammaExponentialKernel(; γ=1.36))

        #Coherence :
        @test isapprox(
            GammaExponentialKernel(; γ=2.0)(sqrt(0.5) * v1, sqrt(0.5) * v2),
            SqExponentialKernel()(v1, v2),
        )
        @test GammaExponentialKernel()(v1, v2) ≈ ExponentialKernel()(v1, v2)
    end
end
