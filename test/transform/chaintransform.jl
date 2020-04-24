@testset "chaintransform" begin
    rng = MersenneTwister(123546)

    P = rand(rng, 3, 2)
    tp = LowRankTransform(P)

    f(x) = sin.(x)
    tf = FunctionTransform(f)

    t = ChainTransform([tp, tf])

    x = ColVecs(randn(rng, 2, 3))
    x′ = map(t, x)

    @test all([t(x[n]) ≈ f(P * x[n]) for n in eachindex(x)])
    @test all([t(x[n]) ≈ x′[n] for n in eachindex(x)])

    @test repr(tp ∘ tf) == "Chain of 2 transforms:\n\t - $(tf) |> $(tp)"
end
