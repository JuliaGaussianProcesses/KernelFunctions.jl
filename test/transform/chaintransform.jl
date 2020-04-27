@testset "chaintransform" begin
    rng = MersenneTwister(123546)

    P = rand(rng, 3, 2)
    tp = LowRankTransform(P)

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
    @test repr(tp ∘ tf) == "Chain of 2 transforms:\n\t - $(tf) |> $(tp)"
end


Base.:∘(t::Transform, tc::ChainTransform) = ChainTransform(vcat(tc.transforms, t))
Base.:∘(tc::ChainTransform, t::Transform) = ChainTransform(vcat(t, tc.transforms))
