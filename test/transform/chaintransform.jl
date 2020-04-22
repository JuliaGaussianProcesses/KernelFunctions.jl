@testset "chaintransform" begin
    rng = MersenneTwister(123546)
    X = rand(rng, 10, 5)
    s = 3.0
    P = rand(rng, 5,10)
    f(x) = sin.(x)

    t = ScaleTransform(s)
    tp = LowRankTransform(P)
    tf = FunctionTransform(f)
    tchain = ChainTransform([t,tp,tf])
    @test all(KernelFunctions.apply(tchain,X,obsdim=2).==f(P*(s*X)))
    @test all(KernelFunctions.apply(tchain,X,obsdim=2).==
                KernelFunctions.apply(tf∘tp∘t,X,obsdim=2))
    @test repr(t∘tf) == "Chain of 2 transforms:\n\t - $(tf) |> $(t)"
end
