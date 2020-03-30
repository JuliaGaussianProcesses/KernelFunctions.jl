@testset "lowranktransform" begin
    dims = (10,5)
    rng = MersenneTwister(123546)
    X = rand(rng, dims...)
    x = rand(rng, dims[1])
    P = rand(rng, 5, 10)

    tp = LowRankTransform(P)
    @test all(KernelFunctions.apply(tp,X,obsdim=2).==P*X)
    @test all(KernelFunctions.apply(tp,x).==P*x)
    @test tp.proj == P
    P2 = rand(5,10)
    KernelFunctions.set!(tp,P2)
    @test all(tp.proj.==P2)
    @test_throws AssertionError KernelFunctions.set!(tp,rand(6,10))
    @test_throws DimensionMismatch KernelFunctions.apply(tp,rand(11,3))
end
