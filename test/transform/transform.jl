@testset "transform" begin
    rng = MersenneTwister(123546)
    x = randn(rng, 8)
    XV = [randn(rng, 5) for _ in 1:6]
    XC = ColVecs(randn(rng, 5, 10))
    XR = RowVecs(randn(rng, 11, 3))
    @testset "IdentityTransform($(typeof(x)))" for x in [x, XV, XC, XR]
        @test IdentityTransform()(x) == x
        @test map(IdentityTransform(), x) == x
    end
    test_ADs(() -> SEKernel() ∘ IdentityTransform())
end
