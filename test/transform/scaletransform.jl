@testset "scaletransform" begin
    X = rand(MersenneTwister(123456), 10, 5)
    s = 3.0

    t = ScaleTransform(s)
    @test all(KernelFunctions.apply(t,X).==s*X)
    s2 = 2.0
    KernelFunctions.set!(t,s2)
    @test all(t.s.==[s2])
    @test isequal(ScaleTransform(s),ScaleTransform(s))
end
