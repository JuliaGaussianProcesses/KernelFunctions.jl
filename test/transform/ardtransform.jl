@testset "ardtransform" begin
    dims = (10,5)
    X = rand(MersenneTwister(123456), dims...)
    s = 3.0
    v1 = vcat(3.0,4.0*ones(dims[2]-1))
    v2 = vcat(3.0,4.0*ones(dims[1]-1))

    vt1 = ARDTransform(v1)
    vt2 = ARDTransform(v2)
    @test all(KernelFunctions.apply(vt1,X,obsdim=1).==v1'.*X)
    @test all(KernelFunctions.apply(vt2,X,obsdim=2).==v2.*X)
    newv1 = rand(5)
    KernelFunctions.set!(vt1,newv1)
    @test all(vt1.v .== newv1)
    @test ARDTransform(s,dims[2]).v == ARDTransform(s*ones(dims[2])).v
    @test_throws DimensionMismatch KernelFunctions.apply(vt1,rand(3,4))
    @test repr(vt1) == "ARD Transform (dims : $(dims[2]))"
end
