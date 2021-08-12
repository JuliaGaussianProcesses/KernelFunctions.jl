@testset "independent" begin
    outdim = 3
    x = KernelFunctions.MOInputIsotopicByOutputs([rand(5) for _ in 1:4], outdim)
    y = KernelFunctions.MOInputIsotopicByOutputs([rand(5) for _ in 1:4], outdim)
    z = KernelFunctions.MOInputIsotopicByOutputs([rand(5) for _ in 1:2], outdim)

    xIF = KernelFunctions.MOInputIsotopicByFeatures(x.x, outdim)
    yIF = KernelFunctions.MOInputIsotopicByFeatures(y.x, outdim)
    zIF = KernelFunctions.MOInputIsotopicByFeatures(z.x, outdim)

    k = IndependentMOKernel(GaussianKernel())
    @test k isa IndependentMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k.kernel isa Kernel

    @test kernelmatrix(k, x, y) == kernelmatrix(k, collect(x), collect(y))

    ## accuracy
    KernelFunctions.TestUtils.test_interface(k, x, y, z)
    KernelFunctions.TestUtils.test_interface(k, xIF, yIF, zIF)

    # in-place
    if VERSION >= v"1.6"
        K = zeros(12, 12)
        kernelmatrix!(K, k, x, y)
        @test K ≈ k.(x, permutedims(y))

        K = zeros(12, 12)
        kernelmatrix!(K, k, xIF, yIF)
        @test K ≈ k.(xIF, permutedims(yIF))
    end

    # type stability (maybe move to test_interface?)
    x2 = MOInput(rand(Float32, 4), 2)
    @test k(x2[1], x2[2]) isa Float32
    @test k(x2[1], x2[1]) isa Float32
    @test eltype(typeof(kernelmatrix(k, x2))) <: Float32

    @test string(k) ==
          "Independent Multi-Output Kernel\n" *
          "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
end
