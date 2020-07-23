@testset "independent" begin
    x = MOInput([rand(5) for _ in 1:4], 3)
    y = MOInput([rand(5) for _ in 1:4], 3)

    k = IndependentMOKernel(GaussianKernel())
    @test k isa IndependentMOKernel
    @test k isa Kernel
    @test k.kernel isa KernelFunctions.BaseKernel
    @test k(x[2], y[2]) isa Real

    @test kernelmatrix(k, x, y) == kernelmatrix(k, collect(x), collect(y))
    @test kernelmatrix(k, x, x) == kernelmatrix(k, x)
    @test string(k) == "Independent Multi-Output Kernel\n\tSquared Exponential Kernel"
end
