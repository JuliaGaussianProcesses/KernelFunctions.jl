@testset "gibbskernel" begin
    rng = MersenneTwister(123456)
    x = randn(rng)
    y = randn(rng)

    k = SqExponentialKernel()

    @test GibbsKernel(x, y, ell) == k(x, y)
end
