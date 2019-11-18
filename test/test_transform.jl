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
    @testset "ScaleTransform" begin
        t = ScaleTransform(s)
        vt1 = ScaleTransform(v1)
        vt2 = ScaleTransform(v2)
        @test all(KernelFunctions.transform(t,X).==s*X)
        @test all(KernelFunctions.transform(vt1,X,1).==v1'.*X)
        @test all(KernelFunctions.transform(vt2,X,2).==v2.*X)
    end
    ## Test LowRankTransform
    @testset "LowRankTransform" begin
        tp = LowRankTransform(P)
        @test all(KernelFunctions.transform(tp,X,2).==P*X)
        @test all(KernelFunctions.transform(tp,x).==P*x)
        @test all(KernelFunctions.params(tp)).==P)
        P2 = rand(5,10)
        KernelFunctions.set!(tp,P2)
        @test all(tp.proj.==P2)
    end
    ## Test FunctionTransform
    @testset "FunctionTransform" begin
        tf = FunctionTransform(f)
        KernelFunctions.transform(tf,X,1)
        @test all(KernelFunctions.transform(tf,X,1).==f(X))
    end
    ## Test SelectTransform
    @testset "SelectTransform" begin
        ts = SelectTransform(sdims)
        @test all(KernelFunctions.transform(ts,X,2).==X[sdims,:])
        @test all(KernelFunctions.transform(ts,x).==x[sdims])
        @test all(KernelFunctions.params(ts).==sdims)
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
        @test all(KernelFunctions.transform(tchain,X,2).==f(P*(s*X)))
        @test all(KernelFunctions.transform(tchain,X,2).==
                    KernelFunctions.transform(tf∘tp∘t,X,2))
    end
end
