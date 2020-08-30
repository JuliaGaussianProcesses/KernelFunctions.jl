@testset "rationalquad" begin
    rng = MersenneTwister(123456)
    x = rand(rng)*2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @testset "RationalQuadraticKernel" begin
        α = 2.0
        k = RationalQuadraticKernel(α=α)
        @test RationalQuadraticKernel(alpha=α).α == [α]
        @test kappa(k,x) ≈ (1.0+x/2.0)^-2
        @test k(v1,v2) ≈ (1.0+norm(v1-v2)^2/2.0)^-2
        @test kappa(RationalQuadraticKernel(α=α),x) == kappa(k,x)
        @test metric(RationalQuadraticKernel()) == SqEuclidean()
        @test metric(RationalQuadraticKernel(α=2.0)) == SqEuclidean()
        @test repr(k) == "Rational Quadratic Kernel (α = $(α))"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(x->RationalQuadraticKernel(alpha=x[1]),[α])
        test_params(k, ([α],))
    end
    @testset "GammaRationalQuadraticKernel" begin
        k = GammaRationalQuadraticKernel()
        @test kappa(k,x) ≈ (1.0+x^2.0/2.0)^-2
        @test k(v1,v2) ≈ (1.0+norm(v1-v2)^4.0/2.0)^-2
        @test kappa(GammaRationalQuadraticKernel(),x) == kappa(k,x)
        a = 1.0 + rand()
        @test GammaRationalQuadraticKernel(alpha=a).α == [a]
        @test repr(k) == "Gamma Rational Quadratic Kernel (α = 2.0, γ = 2.0)"
        #Coherence test
        @test kappa(GammaRationalQuadraticKernel(α=a, γ=1.0), x) ≈ kappa(RationalQuadraticKernel(α=a), x)
        @test metric(GammaRationalQuadraticKernel()) == SqEuclidean()
        @test metric(GammaRationalQuadraticKernel(γ=2.0)) == SqEuclidean()
        @test metric(GammaRationalQuadraticKernel(γ=2.0, α=3.0)) == SqEuclidean()

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        # test_ADs(x->GammaRationalQuadraticKernel(α=x[1], γ=x[2]), [a, 2.0])
        @test_broken "All (problem with power operation)"
        test_params(GammaRationalQuadraticKernel(; α=a, γ=x), ([a], [x]))
    end
end
