@testset "normalizedkernel" begin
    rng = MersenneTwister(123456)
    x = randn(rng)
    y = randn(rng)

    k = 4 * SqExponentialKernel()
    kn = NormalizedKernel(k)
    @test kn(x, y) == k(x, y) / sqrt(k(x, x) * k(y, y))
    @test kn(x, x) ≈ one(x) atol = 1e-5

    # Standardised tests.
    TestUtils.test_interface(kn, Float64)
    test_ADs(x -> NormalizedKernel(exp(x[1]) * SqExponentialKernel()), rand(1))
    test_interface_ad_perf(0.3, StableRNG(123456)) do c
        NormalizedKernel(c * SqExponentialKernel())
    end

    test_params(kn, k)
end
