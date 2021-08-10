@testset "independent" begin
    x = KernelFunctions.MOInputIsotopicByOutputs([rand(5) for _ in 1:4], 3)
    y = KernelFunctions.MOInputIsotopicByOutputs([rand(5) for _ in 1:4], 3)

    k = IndependentMOKernel(GaussianKernel())
    @test k isa IndependentMOKernel
    @test k isa MOKernel
    @test k isa Kernel
    @test k.kernel isa Kernel
    @test k(x[2], y[2]) isa Real

    @test kernelmatrix(k, x, y) == kernelmatrix(k, collect(x), collect(y))
    @test kernelmatrix(k, x, x) == kernelmatrix(k, x)

    x1 = KernelFunctions.MOInputIsotopicByOutputs(rand(5), 3) # Single dim input
    @test k(x1[1], x1[1]) isa Real
    @test kernelmatrix(k, x1) isa Matrix

    ## accuracy
    @test kernelmatrix(k, x, y) ≈ k.(x, permutedims(y))

    x_alt = KernelFunctions.MOInputIsotopicByFeatures(x.x, 3)
    y_alt = KernelFunctions.MOInputIsotopicByFeatures(y.x, 3)
    @test kernelmatrix(k, x_alt, y_alt) ≈ k.(x_alt, permutedims(y_alt))

    # in-place
    K = zeros(12, 12)
    kernelmatrix!(K, k, x, y)
    @test K ≈ k.(x, permutedims(y))

    K = zeros(12, 12)
    kernelmatrix!(K, k, x_alt, y_alt)
    @test K ≈ k.(x_alt, permutedims(y_alt))

    # type stability
    x2 = MOInput(rand(Float32, 4), 2)
    @test k(x2[1], x2[2]) isa Float32
    @test k(x2[1], x2[1]) isa Float32
    @test eltype(typeof(kernelmatrix(k, x2))) <: Float32

    @test string(k) ==
          "Independent Multi-Output Kernel\n" *
          "\tSquared Exponential Kernel (metric = Euclidean(0.0))"
end
