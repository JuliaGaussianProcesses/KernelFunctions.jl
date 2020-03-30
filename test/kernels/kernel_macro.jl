using KernelFunctions
using Test

@testset "Kernel Macro" begin
    @test (@kernel SqExponentialKernel()) isa SqExponentialKernel
    @test (@kernel 3.0*SqExponentialKernel()) isa ScaledKernel{SqExponentialKernel,Float64}
    @test (@kernel 3.0*SqExponentialKernel() l=3.0) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ScaleTransform{Float64}},Float64}
    @test (@kernel 3.0*SqExponentialKernel() 3.0) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ScaleTransform{Float64}},Float64}
    @test (@kernel 3.0*SqExponentialKernel() l=[3.0]) isa ScaledKernel{TransformedKernel{SqExponentialKernel,ARDTransform{Float64,1}},Float64}
    @test (@kernel 3.0*SqExponentialKernel() LowRankTransform(rand(3,2))) isa ScaledKernel{TransformedKernel{SqExponentialKernel,LowRankTransform{Array{Float64,2}}},Float64}
    @test (@kernel (3.0*SqExponentialKernel()+5.0*Matern32Kernel()) 3.0) isa TransformedKernel{KernelSum,ScaleTransform{Float64}}
end
