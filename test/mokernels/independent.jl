@testset "independent" begin
    x = MOInput([rand(5) for _ in 1:4], 3)
    y = MOInput([rand(5) for _ in 1:4], 3)

    k = IndependentMOKernel(GaussianKernel())
    @test k isa IndependentMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k.kernel isa Kernel
    @test k(x[2], y[2]) isa Real

    @test kernelmatrix(k, x, y) == kernelmatrix(k, collect(x), collect(y))
    @test kernelmatrix(k, x, x) == kernelmatrix(k, x)

    x1 = MOInput(rand(5), 3) # Single dim input
    @test k(x1[1], x1[1]) isa Real
    @test kernelmatrix(k, x1) isa Matrix

    # type stability
    x2 = MOInput(rand(Float32, 4), 2)
    @test k(x2[1], x2[2]) isa Float32
    @test k(x2[1], x2[1]) isa Float32
    @test eltype(typeof(kernelmatrix(k, x2))) isa Float32

    @test string(k) ==
          "Independent Multi-Output Kernel\n" *
          "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
end
