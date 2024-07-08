@testset "Periodic Kernel" begin
    x = rand() * 2
    v1 = rand(3)
    v2 = rand(3)
    r = rand(3)
    k = PeriodicKernel(; r=r)
    @test kappa(k, x) ≈ exp(-0.5x)
    @test k(v1, v2) ≈ exp(-0.5 * sum(abs2, sinpi.(v1 - v2) ./ r))
    @test k(v1, v2) == k(v2, v1)
    @test PeriodicKernel(3)(v1, v2) == PeriodicKernel(; r=ones(3))(v1, v2)
    @test isa(PeriodicKernel(r=Float32.(r))(Float32.(v1), Float32.(v2)), Float32) 
    @test repr(k) == "Periodic Kernel, length(r) = $(length(r))"

    # Standardised tests.
    TestUtils.test_interface(PeriodicKernel(; r=[0.9]), Vector{Float64})
    TestUtils.test_interface(PeriodicKernel(; r=[0.9, 0.9]), ColVecs{Float64})
    TestUtils.test_interface(PeriodicKernel(; r=[0.8, 0.7]), RowVecs{Float64})

    test_ADs(r -> PeriodicKernel(; r=exp.(r)), log.(r))
    test_params(k, (r,))
end
