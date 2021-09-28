@testset "gibbskernel" begin
    rng = MersenneTwister(123456)
    x = randn(rng)
    y = randn(rng)

    k = SqExponentialKernel()

    ell(x) = 1.0

    @test GibbsKernel(x, y, ell) == k(x, y)
end
