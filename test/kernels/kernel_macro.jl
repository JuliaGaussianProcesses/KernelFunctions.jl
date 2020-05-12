@testset "Kernel Macro" begin
    @test (@kernel SqExponentialKernel()) isa SqExponentialKernel
    @test_throws ErrorException @kernel sqrt(SqExponentialKernel)
    @test (@kernel 3.0 * SqExponentialKernel()) isa ScaledKernel{SqExponentialKernel,Float64}
    @test (@kernel 3.0 * SqExponentialKernel() l = 3.0) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ScaleTransform{Float64}},Float64}
    # @test (@kernel 3.0 * SqExponentialKernel() 3.0) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ScaleTransform{Float64}},Float64}
    @test (@kernel 3.0 * SqExponentialKernel() l=[3.0]) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ARDTransform{Vector{Float64}}},Float64}
    # @test (@kernel 3.0 * SqExponentialKernel() LinearTransform(rand(3,2))) isa ScaledKernel{TransformedKernel{SqExponentialKernel,LinearTransform{Array{Float64,2}}},Float64}
    # @test (@kernel (3.0 * SqExponen<tialKernel() + 5.0 * Matern32Kernel()) 3.0) isa TransformedKernel{KernelSum,ScaleTransform{Float64}}
end
