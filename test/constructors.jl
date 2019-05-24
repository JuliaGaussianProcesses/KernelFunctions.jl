
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]

## SquaredExponentialKernel
@testset "SquaredExponentialKernel" begin
    @test KernelFunctions.metric(SquaredExponentialKernel(l)) == SqEuclidean()
    @test KernelFunctions.metric(SquaredExponentialKernel(vl)) == WeightedSqEuclidean(vl)
end

SquaredExponentialKernel(l)
