@testset "wiener" begin
    k_1 = WienerKernel(; i=-1)
    @test typeof(k_1) <: WhiteKernel

    k0 = WienerKernel()
    @test k0 isa WienerKernel{0}

    k1 = WienerKernel(; i=1)
    @test k1 isa WienerKernel{1}

    k2 = WienerKernel(; i=2)
    @test k2 isa WienerKernel{2}

    k3 = WienerKernel(; i=3)
    @test k3 isa WienerKernel{3}

    @test_throws ArgumentError WienerKernel(; i=4)
    @test_throws ArgumentError WienerKernel(; i=-2)

    v1 = rand(4)
    v2 = rand(4)

    X = sqrt(sum(abs2, v1))
    Y = sqrt(sum(abs2, v2))
    minXY = min(X, Y)

    @test k0(v1, v2) ≈ minXY
    @test k1(v1, v2) ≈ 1 / 3 * minXY^3 + 1 / 2 * minXY^2 * euclidean(v1, v2)
    @test k2(v1, v2) ≈
        1 / 20 * minXY^5 + 1 / 12 * minXY^3 * euclidean(v1, v2) * (X + Y - 1 / 2 * minXY)
    @test k3(v1, v2) ≈
        1 / 252 * minXY^7 +
          1 / 720 *
          minXY^4 *
          euclidean(v1, v2) *
          (5 * max(X, Y)^2 + 2 * X * Y + 3 * minXY^2)

    # Standardised tests. Requires careful input choice.
    x0 = rand(3)
    x1 = rand(3)
    x2 = rand(2)
    TestUtils.test_interface(k0, x0, x1, x2)
    TestUtils.test_interface(k1, x0, x1, x2)
    TestUtils.test_interface(k2, x0, x1, x2)
    TestUtils.test_interface(k3, x0, x1, x2)
    # test_ADs(()->WienerKernel(i=1))
    @test_broken "No tests passing"
end
