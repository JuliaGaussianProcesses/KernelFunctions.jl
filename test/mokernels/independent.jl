@testset "independent" begin
    x1 = MOInput([rand(5) for _ in 1:4], 3)
    x2 = MOInput([rand(5) for _ in 1:4], 3)

    k = IndependentMOKernel(GaussianKernel())
    @test k isa IndependentMOKernel
    @test k isa Kernel
    @test k.kernel isa KernelFunctions.BaseKernel
    @test k(x1[2], x2[2]) isa Real

    @test kernelmatrix(k, x1, x2) == kernelmatrix(k, collect(x1), collect(x2))
    @test kernelmatrix(k, x1, x1) == kernelmatrix(k, x1)
    @test string(k) == "Independent Multi-Output Kernel\n\tSquared Exponential Kernel"
end
