
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

SquaredExponentialKernel(l)
