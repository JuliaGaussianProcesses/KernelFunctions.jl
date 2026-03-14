@testset "transformedkernel" begin
    rng = MersenneTwister(123456)
    x = rand(rng) * 2
    v1 = rand(rng, 3)
    v2 = rand(rng, 3)

    s = rand(rng)
    v = rand(rng, 3)
    P = rand(rng, 3, 2)
    k = SqExponentialKernel()
    @test k ∘ IdentityTransform() === k

    kt = TransformedKernel(k, ScaleTransform(s))
    ktard = TransformedKernel(k, ARDTransform(v))
    @test kt ∘ IdentityTransform() === kt
    @test ktard ∘ IdentityTransform() === ktard
    @test kt(v1, v2) == (k ∘ ScaleTransform(s))(v1, v2)
    @test kt(v1, v2) ≈ k(s * v1, s * v2) atol = 1e-5
    @test ktard(v1, v2) == (k ∘ ARDTransform(v))(v1, v2)
    @test ktard(v1, v2) == k(v .* v1, v .* v2)
    @test (k ∘ LinearTransform(P') ∘ ScaleTransform(s))(v1, v2) ==
        ((k ∘ LinearTransform(P')) ∘ ScaleTransform(s))(v1, v2) ==
        (k ∘ (LinearTransform(P') ∘ ScaleTransform(s)))(v1, v2)

    @test repr(kt) == repr(k) * "\n\t- " * repr(ScaleTransform(s))

    TestUtils.test_interface(k, Float64)
    TestUtils.test_interface(
        TransformedKernel(ConstantKernel(; c=1.5), FunctionTransform(x -> x * "hi")),
        Vector{String},
    )
    test_ADs(x -> SqExponentialKernel() ∘ ScaleTransform(x[1]), rand(1))
end
