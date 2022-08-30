@testset "matern" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @testset "MaternKernel" begin
        ν = 2.1
        k = MaternKernel(; ν=ν)
        matern(x, ν) = 2^(1 - ν) / gamma(ν) * (sqrt(2ν) * x)^ν * besselk(ν, sqrt(2ν) * x)
        @test MaternKernel(; nu=ν).ν == [ν]
        @test kappa(k, x) ≈ matern(x, ν)
        @test kappa(k, 0.0) == 1.0
        @test metric(MaternKernel()) == Euclidean()
        @test metric(MaternKernel(; ν=2.0)) == Euclidean()
        @test repr(k) == "Matern Kernel (ν = $(ν), metric = Euclidean(0.0))"

        k2 = MaternKernel(; ν=ν, metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(() -> MaternKernel(; nu=ν))

        test_params(k, ([ν],))

        # The performance of this kernel varies quite a lot from method to method, so
        # requires us to specify whether performance tests pass or not.
        @testset "performance ($T)" for T in [
            Vector{Float64},
            ColVecs{Float64,Matrix{Float64}},
            RowVecs{Float64,Matrix{Float64}},
        ]
            xs = example_inputs(StableRNG(123456), Vector{Float64})
            test_interface_ad_perf(
                ν -> MaternKernel(; nu=ν),
                ν,
                xs...;
                passes=(
                    unary=(false, false, false),
                    binary=(false, false, false),
                    diag_unary=(true, false, false),
                    diag_binary=(true, false, false),
                ),
            )
        end
    end
    @testset "Matern32Kernel" begin
        k = Matern32Kernel()
        @test kappa(k, x) ≈ (1 + sqrt(3) * x)exp(-sqrt(3) * x)
        @test k(v1, v2) ≈ (1 + sqrt(3) * norm(v1 - v2))exp(-sqrt(3) * norm(v1 - v2))
        @test kappa(Matern32Kernel(), x) == kappa(k, x)
        @test metric(Matern32Kernel()) == Euclidean()
        @test repr(k) == "Matern 3/2 Kernel (metric = Euclidean(0.0))"

        k2 = Matern32Kernel(; metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(Matern32Kernel)
        test_interface_ad_perf(_ -> Matern32Kernel(), nothing, StableRNG(123456))
    end
    @testset "Matern52Kernel" begin
        k = Matern52Kernel()
        @test kappa(k, x) ≈ (1 + sqrt(5) * x + 5 / 3 * x^2)exp(-sqrt(5) * x)
        @test k(v1, v2) ≈
            (
            1 + sqrt(5) * norm(v1 - v2) + 5 / 3 * norm(v1 - v2)^2
        )exp(-sqrt(5) * norm(v1 - v2))
        @test kappa(Matern52Kernel(), x) == kappa(k, x)
        @test metric(Matern52Kernel()) == Euclidean()
        @test repr(k) == "Matern 5/2 Kernel (metric = Euclidean(0.0))"

        k2 = Matern52Kernel(; metric=WeightedEuclidean(ones(3)))
        @test metric(k2) isa WeightedEuclidean
        @test k2(v1, v2) ≈ k(v1, v2)

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(Matern52Kernel)
        test_interface_ad_perf(_ -> Matern52Kernel(), nothing, StableRNG(123456))
    end
    @testset "Coherence Materns" begin
        @test kappa(MaternKernel(; ν=0.5), x) ≈ kappa(ExponentialKernel(), x)
        @test kappa(MaternKernel(; ν=1.5), x) ≈ kappa(Matern32Kernel(), x)
        @test kappa(MaternKernel(; ν=2.5), x) ≈ kappa(Matern52Kernel(), x)
    end
end
