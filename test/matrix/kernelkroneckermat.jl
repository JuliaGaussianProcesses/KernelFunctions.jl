@testset "kernelkroneckermat" begin
    rng = MersenneTwister(123456)
    k = SqExponentialKernel()
    x = range(0, 1; length=10)
    X = vcat(collect.(Iterators.product(x, x))'...)

    @test all(collect(kernelkronmat(k, collect(x), 2)) .≈ kernelmatrix(k, X; obsdim=1))
    @test all(collect(kernelkronmat(k, [x, x])) .≈ kernelmatrix(k, X; obsdim=1))
    @test_throws AssertionError kernelkronmat(LinearKernel(), collect(x), 2)

    @testset "lazy kernelmatrix" begin
        rng = MersenneTwister(123)

        dims = (in=3, out=2, obs=3)
        r = 1

        A = randn(dims.out, r)
        B = A * transpose(A) + Diagonal(rand(dims.out))

        # XIF = [(rand(dims.in), rand(1:(dims.out))) for i in 1:(dims.obs)]
        x = [rand(dims.in) for _ in 1:2]
        XIF = KernelFunctions.MOInputIsotopicByFeatures(x, dims.out)
        XIO = KernelFunctions.MOInputIsotopicByOutputs(x, dims.out)
        y = [rand(dims.in) for _ in 1:2]
        YIF = KernelFunctions.MOInputIsotopicByFeatures(y, dims.out)
        YIO = KernelFunctions.MOInputIsotopicByOutputs(y, dims.out)

        skernel = GaussianKernel()
        kIndMO = IndependentMOKernel(skernel)

        A = randn(dims.out, r)
        B = A * transpose(A) + Diagonal(rand(dims.out))
        icoregionkernel = IntrinsicCoregionMOKernel(skernel, B)

        function test_kronecker_kernelmatrix(k, x)
            res = kernelmatrix(Kronecker.KroneckerProduct, k, x)
            @test res isa Kronecker.KroneckerProduct
            @test res ≈ @test_deprecated(kronecker_kernelmatrix(k, x))
            @test res ≈ kernelmatrix(k, x)
        end
        function test_kronecker_kernelmatrix(k, x, y)
            res = kernelmatrix(Kronecker.KroneckerProduct, k, x, y)
            @test res isa Kronecker.KroneckerProduct
            @test res ≈ @test_deprecated(kronecker_kernelmatrix(k, x, y))
            @test res ≈ kernelmatrix(k, x, y)
        end

        for k in [kIndMO, icoregionkernel], x in [XIF, XIO]
            test_kronecker_kernelmatrix(k, x)
        end
        for k in [kIndMO, icoregionkernel], (x, y) in ([XIF, YIF], [XIO, YIO])
            test_kronecker_kernelmatrix(k, x, y)
        end

        struct TestMOKernel <: MOKernel end
        @test_throws ArgumentError kernelmatrix(Kronecker.KroneckerProduct, TestMOKernel(), XIF)
        @test_deprecated(@test_throws ArgumentError kronecker_kernelmatrix(TestMOKernel(), XIF))
        @test_throws ArgumentError kernelmatrix(Kronecker.KroneckerProduct, TestMOKernel(), XIF, YIF)
        @test_deprecated(@test_throws ArgumentError kronecker_kernelmatrix(TestMOKernel(), XIF, YIF))
    end
end
