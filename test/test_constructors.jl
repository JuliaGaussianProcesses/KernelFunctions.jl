using KernelFunctions, Test, Distances
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]
s = ScaleTransform(3.0)

## SqExponentialKernel
@testset "SqExponentialKernel" begin
    @test KernelFunctions.metric(SqExponentialKernel(l)) == SqEuclidean()
    @test KernelFunctions.transform(SqExponentialKernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(SqExponentialKernel(vl)) == ScaleTransform(vl)
    @test KernelFunctions.transform(SqExponentialKernel(s)) == s
end

## MaternKernel

@testset "MaternKernel" begin
    @test KernelFunctions.metric(MaternKernel(l)) == Euclidean()
    @test KernelFunctions.metric(Matern32Kernel(l)) == Euclidean()
    @test KernelFunctions.metric(Matern52Kernel(l)) == Euclidean()
    @test KernelFunctions.transform(MaternKernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(Matern32Kernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(Matern52Kernel(l)) == ScaleTransform(l)
    @test KernelFunctions.transform(MaternKernel(vl)) == ScaleTransform(vl)
    @test KernelFunctions.transform(Matern32Kernel(vl)) == ScaleTransform(vl)
    @test KernelFunctions.transform(Matern52Kernel(vl)) == ScaleTransform(vl)
    @test KernelFunctions.transform(MaternKernel(s)) == s
    @test KernelFunctions.transform(Matern32Kernel(s)) == s
    @test KernelFunctions.transform(Matern52Kernel(s)) == s

end
