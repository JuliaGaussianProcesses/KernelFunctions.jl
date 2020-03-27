@testset "FunctionTransform" begin
    rng = MersenneTwister(123546)
    X = rand(rng, 10, 5)
    f(x) = sin.(x)
    tf = FunctionTransform(f)
    KernelFunctions.apply(tf,X,obsdim=1)
    @test all(KernelFunctions.apply(tf,X,obsdim=1).==f(X))
end
