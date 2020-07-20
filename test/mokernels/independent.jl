@testset "ind" begin
    x = MOInput(rand(5), 3)
    y = MOInput(rand(5), 3)

    k = IndependentMOKernel(GaussianKernel(), Matern52Kernel(), GaborKernel())
    @test k isa IndependentMOKernel
    @test k isa KernelFunctions.MOKernel
    @test k.kernels isa Vector{KernelFunctions.BaseKernel}
    @test size(k(x, y)) == (3, 3)

    v1 = [MOInput(rand(5), 3) for _ in 1:5]
    v2 = [MOInput(rand(5), 3) for _ in 1:5]
    # @info kernelmatrix(k, v1, v2)
end