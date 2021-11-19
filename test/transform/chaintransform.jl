@testset "chaintransform" begin
    rng = MersenneTwister(123546)

    P = rand(rng, 3, 2)
    tp = LinearTransform(P)

    f(x) = sin.(x)
    tf = FunctionTransform(f)

    t = ChainTransform([tp, tf])

    # Check composition constructors.
    @test (tf ∘ ChainTransform([tp])).transforms == [tp, tf]
    @test (ChainTransform([tf]) ∘ tp).transforms == [tp, tf]

    # Verify correctness.
    x = ColVecs(randn(rng, 2, 3))
    x′ = map(t, x)

    @test all([t(x[n]) ≈ f(P * x[n]) for n in eachindex(x)])
    @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])

    # Verify printing works as expected.
    @test repr(tp ∘ tf) == "$tp ∘ $tf"
    @test repr(MIME("text/plain"), tp ∘ tf) ==
          "Chain of 2 input transformations:\n   $(tp ∘ tf)"
    test_ADs(
        x -> SEKernel() ∘ (ScaleTransform(exp(x[1])) ∘ ARDTransform(exp.(x[2:4]))),
        randn(rng, 4),
    )
end
