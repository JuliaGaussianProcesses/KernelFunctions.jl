@testset "SelectTransform" begin
    dims = (10,5)
    rng = MersenneTwister(123546)
    X = rand(rng, dims...)
    x = rand(rng, dims[1])
    sdims = [1,2,3]

    ts = SelectTransform(sdims)
    @test all(KernelFunctions.apply(ts,X,obsdim=2).==X[sdims,:])
    @test all(KernelFunctions.apply(ts,x).==x[sdims])
    sdims2 = [2,3,5]
    KernelFunctions.set!(ts,sdims2)
    @test all(ts.select.==sdims2)
end
