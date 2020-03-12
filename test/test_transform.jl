using Test
using KernelFunctions
using Random: seed!

seed!(42)

dims = (10,5)
X = rand(dims...)
x = rand(dims[1])
s = 3.0
v1 = vcat(3.0,4.0*ones(dims[2]-1))
v2 = vcat(3.0,4.0*ones(dims[1]-1))
P = rand(5,10)
sdims = [1,2,3]
f(x) = sin.(x)

@testset "Transform Test" begin
    ## Test Scale Transform
    @testset "IdentityTransform" begin
        @test KernelFunctions.apply(IdentityTransform(),X)==X
    end
    @testset "ScaleTransform" begin
        t = ScaleTransform(s)
        @test all(KernelFunctions.apply(t,X).==s*X)
        s2 = 2.0
        KernelFunctions.set!(t,s2)
        @test all(t.s.==[s2])
        @test isequal(ScaleTransform(s),ScaleTransform(s))
    end
    ## Test ARD Transform
    @testset "ARDTransform" begin
        vt1 = ARDTransform(v1)
        vt2 = ARDTransform(v2)
        @test all(KernelFunctions.apply(vt1,X,obsdim=1).==v1'.*X)
        @test all(KernelFunctions.apply(vt2,X,obsdim=2).==v2.*X)
        newv1 = rand(5)
        KernelFunctions.set!(vt1,newv1)
        @test all(vt1.v .== newv1)
        @test ARDTransform(s,dims[2]).v == ARDTransform(s*ones(dims[2])).v
        @test_throws DimensionMismatch KernelFunctions.apply(vt1,rand(3,4))
    end
    ## Test LowRankTransform
    @testset "LowRankTransform" begin
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
    ## Test FunctionTransform
    @testset "FunctionTransform" begin
        tf = FunctionTransform(f)
        KernelFunctions.apply(tf,X,obsdim=1)
        @test all(KernelFunctions.apply(tf,X,obsdim=1).==f(X))
    end
    ## Test SelectTransform
    @testset "SelectTransform" begin
        ts = SelectTransform(sdims)
        @test all(KernelFunctions.apply(ts,X,obsdim=2).==X[sdims,:])
        @test all(KernelFunctions.apply(ts,x).==x[sdims])
        sdims2 = [2,3,5]
        KernelFunctions.set!(ts,sdims2)
        @test all(ts.select.==sdims2)
    end
    ## Test ChainTransform
    @testset "ChainTransform" begin
        t = ScaleTransform(s)
        tp = LowRankTransform(P)
        tf = FunctionTransform(f)
        tchain = ChainTransform([t,tp,tf])
        @test all(KernelFunctions.apply(tchain,X,obsdim=2).==f(P*(s*X)))
        @test all(KernelFunctions.apply(tchain,X,obsdim=2).==
                    KernelFunctions.apply(tf∘tp∘t,X,obsdim=2))
    end
end
