@testset "piecewisepolynomial" begin
    D = 2
    v1 = rand(D)
    v2 = rand(D)
    maha = Matrix{Float64}(I, D, D)
    degree = 3

    k = PiecewisePolynomialKernel(; degree=degree, dim=D)
    k2 = PiecewisePolynomialKernel{degree}(D)

    @test k2(v1, v2) == k(v1, v2)

    @test_throws MethodError PiecewisePolynomialKernel{4}(maha)
    @test_throws ErrorException PiecewisePolynomialKernel{4}(D)
    @test_throws ErrorException PiecewisePolynomialKernel{degree}(-1)
    @test_throws ErrorException PiecewisePolynomialKernel()
    @test_throws ErrorException PiecewisePolynomialKernel(; degree=degree)

    # default degree
    @test PiecewisePolynomialKernel(; dim=D) isa PiecewisePolynomialKernel{0}

    @test repr(k) ==
          "Piecewise Polynomial Kernel (degree = $(degree), ⌊dim/2⌋ = $(div(D, 2)))"

    # Standardised tests.
    TestUtils.test_interface(k, ColVecs{Float64}; dim_in=2)
    TestUtils.test_interface(k, RowVecs{Float64}; dim_in=2)
    test_ADs(() -> PiecewisePolynomialKernel{degree}(D))

    test_params(k, ())
end
