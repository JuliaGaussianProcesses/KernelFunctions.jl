@testset "rationalquad" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    @testset "RationalQuadraticKernel" begin
        α = 2.0
        k = RationalQuadraticKernel(; α=α)

        @testset "RQ ≈ EQ for large α" begin
            @test isapprox(
                RationalQuadraticKernel(; α=1e9)(v1, v2),
                SqExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @test metric(RationalQuadraticKernel()) == SqEuclidean()
        @test metric(RationalQuadraticKernel(; α=2.0)) == SqEuclidean()
        @test repr(k) == "Rational Quadratic Kernel (α = $(α))"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(x -> RationalQuadraticKernel(; alpha=x[1]), [α])
        test_params(k, ([α],))
    end

    @testset "GammaRationalQuadraticKernel" begin
        k = GammaRationalQuadraticKernel()

        @test repr(k) == "Gamma Rational Quadratic Kernel (α = 2.0, γ = 2.0)"

        @testset "Default GammaRQ ≈ RQ" begin
            @test isapprox(
                GammaRationalQuadraticKernel()(v1, v2), RationalQuadraticKernel()(v1, v2)
            )
            a = 1.0 + rand()
            @test isapprox(
                GammaRationalQuadraticKernel(; α=a)(v1, v2),
                RationalQuadraticKernel(; α=a)(v1, v2),
            )
        end

        @testset "GammaRQ ≈ EQ for large α" begin
            v1 = randn(2)
            v2 = randn(2)
            @test isapprox(
                GammaRationalQuadraticKernel(; α=1e9)(v1, v2),
                SqExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @testset "GammaRQ(γ=1) ≈ Exponential for large α with rescaled inputs" begin
            v1 = randn(4)
            v2 = randn(4)
            @test isapprox(
                GammaRationalQuadraticKernel(; α=1e9, γ=1.0)(2 .* v1, 2 .* v2),
                ExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @testset "GammaRQ ≈ GammaExponential for same γ and large α with rescaled inputs" begin
            v1 = randn(3)
            v2 = randn(3)
            γ = rand() + 0.5
            @test isapprox(
                GammaRationalQuadraticKernel(; α=1e9, γ=γ)(
                    2^(1 / γ) .* v1, 2^(1 / γ) .* v2
                ),
                GammaExponentialKernel(; γ=γ)(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @test metric(GammaRationalQuadraticKernel()) == Euclidean()
        @test metric(GammaRationalQuadraticKernel(; γ=2.0)) == Euclidean()
        @test metric(GammaRationalQuadraticKernel(; γ=2.0, α=3.0)) == Euclidean()

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        a = 1.0 + rand()
        test_ADs(x -> GammaRationalQuadraticKernel(; α=x[1], γ=x[2]), [a, 1 + 0.5 * rand()])
        test_params(GammaRationalQuadraticKernel(; α=a, γ=x), ([a], [x]))
    end
end
