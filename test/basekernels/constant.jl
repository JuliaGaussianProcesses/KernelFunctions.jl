@testset "constant" begin
    @testset "ZeroKernel" begin
        k = ZeroKernel()
        @test eltype(k) == Any
        @test kappa(k, 2.0) == 0.0
        @test binary_op(ZeroKernel()) == KernelFunctions.Delta()
        @test repr(k) == "Zero Kernel"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(ZeroKernel)
    end
    @testset "WhiteKernel" begin
        k = WhiteKernel()
        @test eltype(k) == Any
        @test kappa(k, 1.0) == 1.0
        @test kappa(k, 0.0) == 0.0
        @test EyeKernel == WhiteKernel
        @test binary_op(WhiteKernel()) == KernelFunctions.Delta()
        @test repr(k) == "White Kernel"

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(WhiteKernel)
    end
    @testset "ConstantKernel" begin
        c = 2.0
        k = ConstantKernel(; c=c)
        @test eltype(k) == Any
        @test kappa(k, 1.0) == c
        @test kappa(k, 0.5) == c
        @test binary_op(ConstantKernel()) == KernelFunctions.Delta()
        @test binary_op(ConstantKernel(; c=2.0)) == KernelFunctions.Delta()
        @test repr(k) == "Constant Kernel (c = $(c))"
        test_params(k, ([c],))

        # Standardised tests.
        TestUtils.test_interface(k, Float64)
        test_ADs(c -> ConstantKernel(; c=first(c)), [c])
    end
end
