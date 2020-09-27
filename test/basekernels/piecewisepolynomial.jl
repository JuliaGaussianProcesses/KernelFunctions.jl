@testset "piecewisepolynomial" begin
    D = 2
    v1 = rand(D)
    v2 = rand(D)
    maha = Matrix{Float64}(I, D, D)
    v = 3
    k = PiecewisePolynomialKernel{v}(maha)

    k2 = PiecewisePolynomialKernel(v=v, maha=maha)

    @test k2(v1, v2) ≈ k(v1, v2) atol=1e-5

    @test_throws ErrorException PiecewisePolynomialKernel{4}(maha)

    @test repr(k) == "Piecewise Polynomial Kernel (v = $(v), size(maha) = $(size(maha)))"

    # Standardised tests.
    TestUtils.test_interface(k, ColVecs{Float64}; dim_in=2)
    TestUtils.test_interface(k, RowVecs{Float64}; dim_in=2)
    # test_ADs(maha-> PiecewisePolynomialKernel(v=2, maha = maha), maha)
    @test_broken "Nothing passes (problem with Mahalanobis distance in Distances)"

    test_params(k, (maha,))
end
