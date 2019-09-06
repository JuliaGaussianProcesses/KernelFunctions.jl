using KernelFunctions, Test, Distances
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]

## SquaredExponentialKernel
@testset "SquaredExponentialKernel" begin
    @test KernelFunctions.metric(SquaredExponentialKernel(l)) == SqEuclidean()
    @test KernelFunctions.transform(SquaredExponentialKernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(SquaredExponentialKernel(vl)) == ScaleTransform(vl)
end

@testset "MaternKernel" begin
    @test KernelFunctions.metric(MaternKernel(l)) == Euclidean()
    @test KernelFunctions.metric(MaternKernel(l,1.5)) == Euclidean()
    @test KernelFunctions.metric(MaternKernel(l,2.5)) == Euclidean()
    @test KernelFunctions.transform(MaternKernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(MaternKernel(vl)) == ScaleTransform(vl)
    @test isa(MaternKernel(),Matern32Kernel)
    @test isa(MaternKernel(1.0,1.0),MaternKernel)
    @test isa(MaternKernel(1.0,1.5),Matern32Kernel)
    @test isa(MaternKernel(1.0,2.5),Matern52Kernel)
    @test isa(MaternKernel(1.0,Inf),SquaredExponentialKernel)
end
