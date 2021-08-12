@testset "intrinsiccoregion" begin
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
    z = [rand(dims.in) for _ in 1:3]
    ZIF = KernelFunctions.MOInputIsotopicByFeatures(z, dims.out)
    ZIO = KernelFunctions.MOInputIsotopicByOutputs(z, dims.out)

    kernel = SqExponentialKernel()
    icoregionkernel = IntrinsicCoregionMOKernel(kernel, B)

    icoregionkernel2 = IntrinsicCoregionMOKernel(; kernel=kernel, B=B)
    @test icoregionkernel == icoregionkernel2

    @test icoregionkernel.B == B
    @test icoregionkernel.kernel == kernel
    @test icoregionkernel(XIF[1], XIF[1]) ≈
          B[XIF[1][2], XIF[1][2]] * kernel(XIF[1][1], XIF[1][1])
    @test icoregionkernel(XIF[1], XIF[end]) ≈
          B[XIF[1][2], XIF[end][2]] * kernel(XIF[1][1], XIF[end][1])

    # test convenience function using kronecker product
    @test matrixkernel(icoregionkernel, XIF.x[1], XIF.x[2]) ≈
          icoregionkernel.kernel(XIF.x[1], XIF.x[2]) * icoregionkernel.B

    # kernelmatrix
    KernelFunctions.TestUtils.test_interface(icoregionkernel, XIF, YIF, ZIF)

    KernelFunctions.TestUtils.test_interface(icoregionkernel, XIO, YIO, ZIO)

    KernelFunctions.TestUtils.test_interface(
        icoregionkernel, Vector{Tuple{Float64,Int}}; dim_out=dims.out
    )

    # in-place
    if VERSION >= v"1.6"
        kmsize = dims.out * length(x)
        K = zeros(kmsize, kmsize)
        kernelmatrix!(K, icoregionkernel, XIF, XIF)
        @test K ≈ icoregionkernel.(XIF, permutedims(XIF))

        K = zeros(kmsize, kmsize)
        kernelmatrix!(K, icoregionkernel, XIO, XIO)
        @test K ≈ icoregionkernel.(XIO, permutedims(XIO))
    end

    test_ADs(icoregionkernel; dims=dims)

    @test string(icoregionkernel) ==
          string("Intrinsic Coregion Kernel: ", kernel, " with ", dims.out, " outputs")
end
