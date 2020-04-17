@testset "constant" begin
    @testset "ZeroKernel" begin
        k = ZeroKernel()
        @test eltype(k) == Any
        @test kappa(k,2.0) == 0.0
        @test KernelFunctions.metric(ZeroKernel()) == KernelFunctions.Delta()
        @test repr(k) == "Zero Kernel"
    end
    @testset "WhiteKernel" begin
        k = WhiteKernel()
        @test eltype(k) == Any
        @test kappa(k,1.0) == 1.0
        @test kappa(k,0.0) == 0.0
        @test EyeKernel == WhiteKernel
        @test metric(WhiteKernel()) == KernelFunctions.Delta()
        @test repr(k) == "White Kernel"
    end
    @testset "ConstantKernel" begin
        c = 2.0
        k = ConstantKernel(c=c)
        @test eltype(k) == Any
        @test kappa(k,1.0) == c
        @test kappa(k,0.5) == c
        @test metric(ConstantKernel()) == KernelFunctions.Delta()
        @test metric(ConstantKernel(c=2.0)) == KernelFunctions.Delta()
        @test repr(k) == "Constant Kernel (c = $(c))"
    end
end
