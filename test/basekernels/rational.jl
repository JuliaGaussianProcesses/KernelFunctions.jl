@testset "rational.jl" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    @testset "RationalKernel" begin
        α = rand()
        k = RationalKernel(; α=α)

        @testset "RationalKernel ≈ Exponential for large α" begin
            @test isapprox(
                RationalKernel(; α=1e9)(v1, v2),
                ExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @test metric(RationalKernel()) == Euclidean()
        @test metric(RationalKernel(; α=α)) == Euclidean()
        @test repr(k) == "Rational Kernel (α = $(α), metric = Euclidean(0.0))"

        k2 = RationalKernel(; α=α, metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(x -> RationalKernel(; alpha=exp(x[1])), [α])
        test_params(k, ([α],))
        test_interface_ad_perf(α -> RationalKernel(; alpha=α), α, StableRNG(123456))
    end

    @testset "RationalQuadraticKernel" begin
        α = rand()
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
        @test metric(RationalQuadraticKernel(; α=α)) == SqEuclidean()
        @test repr(k) == "Rational Quadratic Kernel (α = $(α), metric = Euclidean(0.0))"

        k2 = RationalQuadraticKernel(; α=α, metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        # test_ADs(x -> RationalQuadraticKernel(; alpha=exp(x[1])), [α])
        test_params(k, ([α],))
        test_interface_ad_perf(α, StableRNG(123456)) do α
            RationalQuadraticKernel(; alpha=α)
        end

        # Check correctness and performance with non-Euclidean metrics.
        TestUtils.test_interface(
            RationalQuadraticKernel(; alpha=α, metric=WeightedEuclidean([1.0, 2.0])),
            ColVecs{Float64},
        )
        TestUtils.test_interface(
            RationalQuadraticKernel(; alpha=α, metric=WeightedEuclidean([1.0, 2.0])),
            RowVecs{Float64},
        )
        types = [ColVecs{Float64,Matrix{Float64}}, RowVecs{Float64,Matrix{Float64}}]
        test_interface_ad_perf(α, StableRNG(123456)) do α
            RationalQuadraticKernel(; alpha=α, metric=KernelFunctions.DotProduct())
        end
    end

    @testset "GammaRationalKernel" begin
        k = GammaRationalKernel()

        @test repr(k) == "Gamma Rational Kernel (α = 2.0, γ = 1.0, metric = Euclidean(0.0))"

        @testset "GammaRational (γ=2) ≈ RQ with rescaled inputs" begin
            @test isapprox(
                GammaRationalKernel(; γ=2)(v1 ./ sqrt(2), v2 ./ sqrt(2)),
                RationalQuadraticKernel()(v1, v2),
            )
            a = 1 + rand()
            @test isapprox(
                GammaRationalKernel(; α=a, γ=2)(v1 ./ sqrt(2), v2 ./ sqrt(2)),
                RationalQuadraticKernel(; α=a)(v1, v2),
            )
        end

        @testset "GammaRational (γ=2) ≈ EQ for large α with rescaled inputs" begin
            v1 = randn(2)
            v2 = randn(2)
            @test isapprox(
                GammaRationalKernel(; α=1e9, γ=2)(v1 ./ sqrt(2), v2 ./ sqrt(2)),
                SqExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @testset "Default GammaRational ≈ Rational" begin
            @test isapprox(GammaRationalKernel()(v1, v2), RationalKernel()(v1, v2))
            a = 1 + rand()
            @test isapprox(
                GammaRationalKernel(; α=a)(v1, v2), RationalKernel(; α=a)(v1, v2)
            )
        end

        @testset "Default GammaRational ≈ Exponential for large α" begin
            v1 = randn(4)
            v2 = randn(4)
            @test isapprox(
                GammaRationalKernel(; α=1e9)(v1, v2),
                ExponentialKernel()(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @testset "GammaRational ≈ GammaExponential for same γ and large α" begin
            v1 = randn(3)
            v2 = randn(3)
            γ = rand() + 0.5
            @test isapprox(
                GammaRationalKernel(; α=1e9, γ=γ)(v1, v2),
                GammaExponentialKernel(; γ=γ)(v1, v2);
                atol=1e-6,
                rtol=1e-6,
            )
        end

        @test metric(GammaRationalKernel()) == Euclidean()
        @test metric(GammaRationalKernel(; γ=2.0)) == Euclidean()
        @test metric(GammaRationalKernel(; γ=2.0, α=3.0)) == Euclidean()

        k2 = GammaRationalKernel(; metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        a = 1.0 + rand()
        test_ADs(x -> GammaRationalKernel(; α=x[1], γ=x[2]), [a, 1 + 0.5 * rand()])
        test_params(GammaRationalKernel(; α=a, γ=x), ([a], [x]))
        test_interface_ad_perf((2.0, 1.5), StableRNG(123456)) do θ
            GammaRationalKernel(; α=θ[1], γ=θ[2])
        end
    end
end
