@testset "gibbskernel" begin
    x = randn(2)
    y = randn(2)

    # generate random number of standard SqExponentialKernel lengthscale
    ell = randn(1)[1]

    # this is the gibbs lengthscale function.
    # to check that we can recover the stationary SqExponentialKernel
    # with a constant lengthscale we just set this function
    # equal to a constant.
    l_func(x) = ell

    # in order to compare to the Gibbs kernel we compute the
    # equivalent lengthscale l(x)^2 + l(y)^2.
    # See the denominator of the exponential term in the gibbs kernel.
    lengthscale = hypot(l_func(x), l_func(y))

    # create a SqExponentialKernel with a lengthscale
    k = with_lengthscale(SqExponentialKernel(), lengthscale)

    # create a gibbs kernel with our constant lengthscale function
    k_gibbs = GibbsKernel(l_func)

    # check they are equal
    @test k_gibbs(x, y) == k(x, y)
end