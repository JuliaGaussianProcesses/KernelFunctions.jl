@testset "gibbskernel" begin
    x = 1.0
    y = 1.0

    k = SqExponentialKernel()

    ell(x) = 1
    k_gibbs = GibbsKernel(ell)

    @test k_gibbs(x, y) == k(x, y)
end
