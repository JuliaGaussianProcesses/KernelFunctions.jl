@testset "matern" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)
    @testset "MaternKernel" begin
        ν = 2.0
        k = MaternKernel(; ν=ν)
        matern(x, ν) = 2^(1 - ν) / gamma(ν) * (sqrt(2ν) * x)^ν * besselk(ν, sqrt(2ν) * x)
        @test MaternKernel(; nu=ν).ν == [ν]
        @test kappa(k, x) ≈ matern(x, ν)
        @test kappa(k, 0.0) == 1.0
        @test kappa(MaternKernel(; ν=ν), x) == kappa(k, x)
        @test binary_op(MaternKernel()) == Euclidean()
        @test binary_op(MaternKernel(; ν=2.0)) == Euclidean()
        @test repr(k) == "Matern Kernel (ν = $(ν))"
        # test_ADs(x->MaternKernel(nu=first(x)),[ν])
        @test_broken "All fails (because of logabsgamma for ForwardDiff and ReverseDiff and because of nu for Zygote)"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_params(k, ([ν],))
    end
    @testset "Matern32Kernel" begin
        k = Matern32Kernel()
        @test kappa(k, x) ≈ (1 + sqrt(3) * x)exp(-sqrt(3) * x)
        @test k(v1, v2) ≈ (1 + sqrt(3) * norm(v1 - v2))exp(-sqrt(3) * norm(v1 - v2))
        @test kappa(Matern32Kernel(), x) == kappa(k, x)
        @test binary_op(Matern32Kernel()) == Euclidean()
        @test repr(k) == "Matern 3/2 Kernel"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(Matern32Kernel)
    end
    @testset "Matern52Kernel" begin
        k = Matern52Kernel()
        @test kappa(k, x) ≈ (1 + sqrt(5) * x + 5 / 3 * x^2)exp(-sqrt(5) * x)
        @test k(v1, v2) ≈
              (
            1 + sqrt(5) * norm(v1 - v2) + 5 / 3 * norm(v1 - v2)^2
        )exp(-sqrt(5) * norm(v1 - v2))
        @test kappa(Matern52Kernel(), x) == kappa(k, x)
        @test binary_op(Matern52Kernel()) == Euclidean()
        @test repr(k) == "Matern 5/2 Kernel"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(Matern52Kernel)
    end
    @testset "Coherence Materns" begin
        @test kappa(MaternKernel(; ν=0.5), x) ≈ kappa(ExponentialKernel(), x)
        @test kappa(MaternKernel(; ν=1.5), x) ≈ kappa(Matern32Kernel(), x)
        @test kappa(MaternKernel(; ν=2.5), x) ≈ kappa(Matern52Kernel(), x)
    end
end
