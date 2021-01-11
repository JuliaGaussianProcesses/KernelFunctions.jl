@testset "piecewisepolynomial" begin
    D = 2
    v1 = rand(D)
    v2 = rand(D)
    maha = Matrix{Float64}(I, D, D)
    v = 3

    k = PiecewisePolynomialKernel(; v=v, d=D)
    k2 = PiecewisePolynomialKernel{v}(D)
    k3 = @test_deprecated PiecewisePolynomialKernel{v}(maha)
    k4 = @test_deprecated PiecewisePolynomialKernel(; v=v, maha=maha)

    @test k2(v1, v2) == k(v1, v2)
    @test k3(v1, v2) ≈ k(v1, v2)
    @test k4(v1, v2) ≈ k(v1, v2)

    @test_throws ErrorException PiecewisePolynomialKernel{4}(maha)
    @test_throws ErrorException PiecewisePolynomialKernel{4}(D)
    @test_throws ErrorException PiecewisePolynomialKernel{v}(-1)

    @test repr(k) == "Piecewise Polynomial Kernel (v = $(v), ⌊d/2⌋ = $(div(D, 2)))"

    # Standardised tests.
    TestUtils.test_interface(k, ColVecs{Float64}; dim_in=2)
    TestUtils.test_interface(k, RowVecs{Float64}; dim_in=2)
    test_ADs(() -> PiecewisePolynomialKernel{v}(D))

    test_params(k, ())
end
