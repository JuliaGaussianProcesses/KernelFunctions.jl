@testset "zygote_adjoints" begin

    rng = MersenneTwister(123456)
    x = rand(rng, 5)
    y = rand(rng, 5)
    r = rand(rng, 5)

    gzeucl = gradient(Val(:Zygote), xy -> evaluate(Euclidean(), xy[1], xy[2]), [x,y])
    gzsqeucl =  gradient(Val(:Zygote), xy -> evaluate(SqEuclidean(), xy[1], xy[2]), [x,y])
    gzdotprod = gradient(Val(:Zygote), xy -> evaluate(KernelFunctions.DotProduct(), xy[1], xy[2]), [x,y])
    gzdelta = gradient(Val(:Zygote), xy -> evaluate(KernelFunctions.Delta(), xy[1], xy[2]), [x,y])
    gzsinus = gradient(Val(:Zygote), xy -> evaluate(KernelFunctions.Sinus(r), xy[1], xy[2]), [x,y])

    gfeucl = gradient(Val(:FiniteDiff), xy -> evaluate(Euclidean(), xy[1], xy[2]), [x,y])
    gfsqeucl = gradient(Val(:FiniteDiff), xy -> evaluate(SqEuclidean(), xy[1], xy[2]), [x,y])
    gfdotprod = gradient(Val(:FiniteDiff), xy -> evaluate(KernelFunctions.DotProduct(), xy[1], xy[2]), [x,y])
    gfdelta = gradient(Val(:FiniteDiff), xy -> evaluate(KernelFunctions.Delta(), xy[1], xy[2]), [x,y])
    gfsinus = gradient(Val(:FiniteDiff), xy -> evaluate(KernelFunctions.Sinus(r), xy[1], xy[2]), [x,y])


    @test all(gzeucl .≈ gfeucl)
    @test all(gzsqeucl .≈ gfsqeucl)
    @test all(gzdotprod .≈ gfdotprod)
    @test all(gzdelta .≈ gfdelta)
    @test all(gzsinus .≈ gfsinus)
end
