@testset "intrinsiccoregion" begin
    rng = MersenneTwister(123)

    dims = (in=3, out=2, obs=3)
    r = 1

    A = randn(dims.out, r)
    B = A * transpose(A) + Diagonal(rand(dims.out))

    # X = [(rand(dims.in), rand(1:(dims.out))) for i in 1:(dims.obs)]
    x = [rand(dims.in) for _ in 1:2]
    X = KernelFunctions.MOInputIsotopicByFeatures(x, dims.out)

    kernel = SqExponentialKernel()
    icoregionkernel = IntrinsicCoregionMOKernel(; kernel=kernel, B=B)

    icoregionkernel2 = IntrinsicCoregionMOKernel(kernel, B)
    @test icoregionkernel == icoregionkernel2

    @test icoregionkernel.B == B
    @test icoregionkernel.kernel == kernel
    @test icoregionkernel(X[1], X[1]) ≈ B[X[1][2], X[1][2]] * kernel(X[1][1], X[1][1])
    @test icoregionkernel(X[1], X[end]) ≈ B[X[1][2], X[end][2]] * kernel(X[1][1], X[end][1])

    # test convenience function using kronecker product
    @test matrixkernel(icoregionkernel, X.x[1], X.x[2]) ≈
          icoregionkernel.kernel(X.x[1], X.x[2]) * icoregionkernel.B

    # kernelmatrix
    @test kernelmatrix(icoregionkernel, X) ≈ icoregionkernel.(X, permutedims(X))

    X_alt = KernelFunctions.MOInputIsotopicByOutputs(x, dims.out)
    @test kernelmatrix(icoregionkernel, X_alt) ≈ icoregionkernel.(X_alt, permutedims(X_alt))

    KernelFunctions.TestUtils.test_interface(
        icoregionkernel, Vector{Tuple{Float64,Int}}; dim_out=dims.out
    )

    # in-place
    kmsize = dims.out * length(x)
    K = zeros(kmsize, kmsize)
    kernelmatrix!(K, icoregionkernel, X, X)
    @test K ≈ icoregionkernel.(X, permutedims(X))

    K = zeros(kmsize, kmsize)
    kernelmatrix!(K, icoregionkernel, X_alt, X_alt)
    @test K ≈ icoregionkernel.(X_alt, permutedims(X_alt))

    test_ADs(icoregionkernel; dims=dims)

    @test string(icoregionkernel) ==
          string("Intrinsic Coregion Kernel: ", kernel, " with ", dims.out, " outputs")
end
