@testset "wiener" begin
    k_1 = WienerKernel(i=-1)
    @test typeof(k_1) <: WhiteKernel

    k0 = WienerKernel()
    @test typeof(k0) <: WienerKernel{0}

    k1 = WienerKernel(i=1)
    @test typeof(k1) <: WienerKernel{1}

    k2 = WienerKernel(i=2)
    @test typeof(k2) <: WienerKernel{2}

    k3 = WienerKernel(i=3)
    @test typeof(k3) <: WienerKernel{3}

    @test_throws AssertionError WienerKernel(i=4)
    @test_throws AssertionError WienerKernel(i=-2)

    # kernelmatrix tests
    m1 = rand(3,4)
    m2 = rand(3,4)
    @test kernelmatrix(k0, m1, m1) ≈ kernelmatrix(k0, m1) atol=1e-5

    K = zeros(4,4)
    kernelmatrix!(K,k0,m1,m2)
    @test K ≈ kernelmatrix(k0, m1, m2) atol=1e-5

    V = zeros(4)
    kerneldiagmatrix!(V,k0,m1)
    @test V ≈ kerneldiagmatrix(k0,m1) atol=1e-5

    x1 = rand()
    x2 = rand()
    @test kernelmatrix(k0, x1*ones(1,1), x2*ones(1,1))[1] ≈ k0(x1, x2) atol=1e-5
    @test kernelmatrix(k1, x1*ones(1,1), x2*ones(1,1))[1] ≈ k1(x1, x2) atol=1e-5
    @test kernelmatrix(k2, x1*ones(1,1), x2*ones(1,1))[1] ≈ k2(x1, x2) atol=1e-5
    @test kernelmatrix(k3, x1*ones(1,1), x2*ones(1,1))[1] ≈ k3(x1, x2) atol=1e-5
end
