@testset "piecewisepolynomial" begin
    v1 = rand(3)
    v2 = rand(3)
    m1 = rand(3, 4)
    m2 = rand(3, 4)
    maha = ones(3, 3)
    v = 3
    k = PiecewisePolynomialKernel{v}(maha)

    k2 = PiecewisePolynomialKernel(v=v, maha=maha)

    @test k2(v1, v2) ≈ k(v1, v2) atol=1e-5

    @test_throws ErrorException PiecewisePolynomialKernel{4}(maha)

    @test repr(k) == "Piecewise Polynomial Kernel (v = $(v), size(maha) = $(size(maha)))"

    # Standardised tests.
    TestUtils.test_interface(k, Float64)
    # test_ADs(maha-> PiecewisePolynomialKernel(v=2, maha = maha), maha)
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"

    test_params(k, (maha,))
end
