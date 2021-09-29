@testset "gibbskernel" begin
    x = 1.0
    y = 2.0

    k = SqExponentialKernel()

    ell(x) = 1.0

    @test GibbsKernel(x, y, ell) == k(x, y)
end
