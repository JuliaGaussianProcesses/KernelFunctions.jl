@testset "scaledkernel" begin
    rng = MersenneTwister(123456)
    x = randn(rng)
    y = randn(rng)
    s = rand(rng) + 1e-3

    k = SqExponentialKernel()
    ks = ScaledKernel(k, s)
    @test ks(x, y) == s * k(x, y)
    @test ks(x, y) == (s * k)(x, y)

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    test_ADs(x -> exp(x[1]) * SqExponentialKernel(), rand(1))

    test_params(s * k, (k, [s]))
end
