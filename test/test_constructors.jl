using KernelFunctions, Test, Distances
#TODO Test metric weights for ARD, test equivalency for different constructors,
# test type conversion
l = 2.0
vl = [l,l]
s = ScaleTransform(l)

## Add tests for Transformed Kernel and Scaled Kernel

# ## SqExponentialKernel
# @testset "Exponential" begin
#     @test KernelFunctions.metric(ExponentialKernel()) == Euclidean()
#     @test KernelFunctions.metric(SqExponentialKernel()) == SqEuclidean()
#     @test KernelFunctions.metric(GammaExponentialKernel()) == SqEuclidean()
#     @test KernelFunctions.metric(GammaExponentialKernel(γ=2.0)) == SqEuclidean()
# end

# ## MaternKernel
# @testset "MaternKernel" begin
#     @test KernelFunctions.metric(MaternKernel()) == Euclidean()
#     @test KernelFunctions.metric(MaternKernel(ν=2.0)) == Euclidean()
#     @test KernelFunctions.metric(Matern32Kernel()) == Euclidean()
#     @test KernelFunctions.metric(Matern52Kernel()) == Euclidean()
# end

# @testset "Exponentiated" begin
#     @test KernelFunctions.metric(ExponentiatedKernel()) == KernelFunctions.DotProduct()
# end

# @testset "Constant" begin
#     @test KernelFunctions.metric(ConstantKernel()) == KernelFunctions.Delta()
#     @test KernelFunctions.metric(ConstantKernel(c=2.0)) == KernelFunctions.Delta()
#     @test KernelFunctions.metric(WhiteKernel()) == KernelFunctions.Delta()
#     @test KernelFunctions.metric(ZeroKernel()) == KernelFunctions.Delta()
# end

# @testset "Polynomial" begin
#     @test KernelFunctions.metric(LinearKernel()) == KernelFunctions.DotProduct()
#     @test KernelFunctions.metric(LinearKernel(c=2.0)) == KernelFunctions.DotProduct()
#     @test KernelFunctions.metric(PolynomialKernel()) == KernelFunctions.DotProduct()
#     @test KernelFunctions.metric(PolynomialKernel(d=3.0)) == KernelFunctions.DotProduct()
#     @test KernelFunctions.metric(PolynomialKernel(d=3.0,c=2.0)) == KernelFunctions.DotProduct()
# end

# @testset "RationalQuadratic" begin
#     @test metric(RationalQuadraticKernel()) == SqEuclidean()
#     @test metric(RationalQuadraticKernel(α=2.0)) == SqEuclidean()
#     @test metric(GammaRationalQuadraticKernel()) == SqEuclidean()
#     @test metric(GammaRationalQuadraticKernel(γ=2.0)) == SqEuclidean()
#     @test metric(GammaRationalQuadraticKernel(γ=2.0,α=3.0)) == SqEuclidean()
# end
