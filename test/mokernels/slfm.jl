@testset "slfm" begin
    x1 = MOInput([rand(5) for _ in 1:4], 2)
    x2 = MOInput([rand(5) for _ in 1:4], 2)

    k = LatentFactorMOKernel(
        [MaternKernel(), SqExponentialKernel(), FBMKernel()],
        [ExponentialKernel(), PeriodicKernel(5)],
        rand(2, 3)
    )
    @test k isa LatentFactorMOKernel
    @test k isa Kernel
    @info k.g[2]
    @test k(x1[2], x2[2]) isa Real

    @test kernelmatrix(k, x1, x2) == kernelmatrix(k, collect(x1), collect(x2))
    @test kernelmatrix(k, x1, x1) == kernelmatrix(k, x1)

end
