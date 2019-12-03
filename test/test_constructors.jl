using KernelFunctions, Test, Distances
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]
s = ScaleTransform(l)

## SqExponentialKernel
@testset "SqExponentialKernel" begin
    @test KernelFunctions.metric(SqExponentialKernel(l)) == SqEuclidean()
    @test isequal(transform(SqExponentialKernel(l)),s)
    @test KernelFunctions.transform(SqExponentialKernel(vl)) == ARDTransform(vl)
    @test isequal(KernelFunctions.transform(SqExponentialKernel(s)),s)
end

## MaternKernel
@testset "MaternKernel" begin
    @test KernelFunctions.metric(MaternKernel(l)) == Euclidean()
    @test KernelFunctions.metric(Matern32Kernel(l)) == Euclidean()
    @test KernelFunctions.metric(Matern52Kernel(l)) == Euclidean()
    @test isequal(KernelFunctions.transform(MaternKernel(l)),s)
    @test isequal(KernelFunctions.transform(Matern32Kernel(l)),s)
    @test isequal(KernelFunctions.transform(Matern52Kernel(l)),s)
    @test KernelFunctions.transform(MaternKernel(vl)) == ARDTransform(vl)
    @test KernelFunctions.transform(Matern32Kernel(vl)) == ARDTransform(vl)
    @test KernelFunctions.transform(Matern52Kernel(vl)) == ARDTransform(vl)
    @test KernelFunctions.transform(MaternKernel(s)) == s
    @test KernelFunctions.transform(Matern32Kernel(s)) == s
    @test KernelFunctions.transform(Matern52Kernel(s)) == s
end
