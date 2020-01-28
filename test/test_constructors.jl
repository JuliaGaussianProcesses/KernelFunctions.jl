using KernelFunctions, Test, Distances
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]
s = ScaleTransform(l)

## Add tests for Transformed Kernel and Scaled Kernel

## SqExponentialKernel
@testset "Exponential" begin
    @test KernelFunctions.metric(ExponentialKernel()) == Euclidean()
    @test KernelFunctions.metric(SqExponentialKernel()) == SqEuclidean()
    @test KernelFunctions.metric(GammaExponentialKernel()) == SqEuclidean()
    @test KernelFunctions.metric(GammaExponentialKernel(2.0)) == SqEuclidean()
    # @test isequal(transform(SqExponentialKernel(l)),s)
    # @test KernelFunctions.transform(SqExponentialKernel(vl)) == ARDTransform(vl)
    # @test isequal(KernelFunctions.transform(SqExponentialKernel(s)),s)
end

## MaternKernel
@testset "MaternKernel" begin
    @test KernelFunctions.metric(MaternKernel()) == Euclidean()
    @test KernelFunctions.metric(MaternKernel(2.0)) == Euclidean()
    @test KernelFunctions.metric(Matern32Kernel()) == Euclidean()
    @test KernelFunctions.metric(Matern52Kernel()) == Euclidean()
    # @test isequal(KernelFunctions.transform(MaternKernel(l)),s)
    # @test isequal(KernelFunctions.transform(Matern32Kernel(l)),s)
    # @test isequal(KernelFunctions.transform(Matern52Kernel(l)),s)
    # @test KernelFunctions.transform(MaternKernel(vl)) == ARDTransform(vl)
    # @test KernelFunctions.transform(Matern32Kernel(vl)) == ARDTransform(vl)
    # @test KernelFunctions.transform(Matern52Kernel(vl)) == ARDTransform(vl)
    # @test KernelFunctions.transform(MaternKernel(s)) == s
    # @test KernelFunctions.transform(Matern32Kernel(s)) == s
    # @test KernelFunctions.transform(Matern52Kernel(s)) == s
end

@testset "Exponentiated" begin
    @test KernelFunctions.metric(ExponentiatedKernel()) == KernelFunctions.DotProduct()
end

@testset "Constant" begin
    @test KernelFunctions.metric(ConstantKernel()) == KernelFunctions.Delta()
    @test KernelFunctions.metric(ConstantKernel(2.0)) == KernelFunctions.Delta()
    @test KernelFunctions.metric(WhiteKernel()) == KernelFunctions.Delta()
    @test KernelFunctions.metric(ZeroKernel()) == KernelFunctions.Delta()
end

@testset "Polynomial" begin
    @test KernelFunctions.metric(LinearKernel()) == KernelFunctions.DotProduct()
    @test KernelFunctions.metric(LinearKernel(2.0)) == KernelFunctions.DotProduct()
    @test KernelFunctions.metric(PolynomialKernel()) == KernelFunctions.DotProduct()
    @test KernelFunctions.metric(PolynomialKernel(3.0)) == KernelFunctions.DotProduct()
    @test KernelFunctions.metric(PolynomialKernel(3.0,2.0)) == KernelFunctions.DotProduct()
end

@testset "RationalQuadratic" begin
    @test KernelFunctions.metric(RationalQuadraticKernel()) == SqEuclidean()
    @test KernelFunctions.metric(RationalQuadraticKernel(2.0)) == SqEuclidean()
    @test KernelFunctions.metric(GammaRationalQuadraticKernel()) == SqEuclidean()
    @test KernelFunctions.metric(GammaRationalQuadraticKernel(2.0)) == SqEuclidean()
    @test KernelFunctions.metric(GammaRationalQuadraticKernel(2.0,3.0)) == SqEuclidean()
end
