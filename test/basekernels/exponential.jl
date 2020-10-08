@testset "exponential" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @testset "SqExponentialKernel" begin
        k = SqExponentialKernel()
        @test kappa(k,x) ≈ exp(-x / 2)
        @test k(v1,v2) ≈ exp(-norm(v1-v2)^2 / 2)
        @test kappa(SqExponentialKernel(),x) == kappa(k,x)
        @test metric(SqExponentialKernel()) == SqEuclidean()
        @test RBFKernel == SqExponentialKernel
        @test GaussianKernel == SqExponentialKernel
        @test SEKernel == SqExponentialKernel
        @test repr(k) == "Squared Exponential Kernel"
        @test KernelFunctions.iskroncompatible(k) == true

        # Standardised tests.
        TestUtils.test_interface(k)
        test_ADs(SEKernel)
    end
    @testset "ExponentialKernel" begin
        k = ExponentialKernel()
        @test kappa(k,x) ≈ exp(-x)
        @test k(v1,v2) ≈ exp(-norm(v1-v2))
        @test kappa(ExponentialKernel(),x) == kappa(k,x)
        @test metric(ExponentialKernel()) == Euclidean()
        @test repr(k) == "Exponential Kernel"
        @test LaplacianKernel == ExponentialKernel
        @test KernelFunctions.iskroncompatible(k) == true

        # Standardised tests.
        TestUtils.test_interface(k)
        test_ADs(ExponentialKernel)
    end
    @testset "GammaExponentialKernel" begin
        γ = 2.0
        k = GammaExponentialKernel(γ=γ)
        @test k(v1, v2) ≈ exp(-norm(v1 - v2)^γ)
        @test kappa(GammaExponentialKernel(), x) == kappa(k, x)
        @test GammaExponentialKernel(gamma=γ).γ == [γ]
        @test metric(GammaExponentialKernel()) == Euclidean()
        @test metric(GammaExponentialKernel(γ=2.0)) == Euclidean()
        @test repr(k) == "Gamma Exponential Kernel (γ = $(γ))"
        @test KernelFunctions.iskroncompatible(k) == true

        test_ADs(γ -> GammaExponentialKernel(gamma=first(γ)), [γ])
        test_params(k, ([γ],))
        TestUtils.test_interface(GammaExponentialKernel(γ=1.36))

        #Coherence :
        @test isapprox(
            GammaExponentialKernel(γ=2.0)(sqrt(0.5) * v1, sqrt(0.5) * v2),
            SqExponentialKernel()(v1,v2),
        )
        @test GammaExponentialKernel(γ=1.0)(v1, v2) ≈ ExponentialKernel()(v1, v2)
    end
end
