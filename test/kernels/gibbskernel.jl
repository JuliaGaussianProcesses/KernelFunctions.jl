@testset "gibbskernel" begin
    x = randn()
    y = randn()

    # this is the gibbs lengthscale function.
    ell(x) = exp(sum(sin, x))
    # create a gibbs kernel with our specific lengthscale function
    k_gibbs = GibbsKernel(ell)

    @test k_gibbs(x, y) ≈
          sqrt((2 * ell(x) * ell(y)) / (ell(x)^2 + ell(y)^2)) *
          exp(-(x - y)^2 / (ell(x)^2 + ell(y)^2))

end