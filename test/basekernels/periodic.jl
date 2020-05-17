@testset "Periodic Kernel" begin
    x = rand()*2; v1 = rand(3); v2 = rand(3);
    r = rand(3)
    k = PeriodicKernel(r = r)
    @test kappa(k, x) ≈ exp(-0.5x)
    @test k(v1, v2) ≈ exp(-0.5 * sum(abs2, sinpi.(v1 - v2) ./ r))
    @test k(v1, v2) == k(v2, v1)
    @test PeriodicKernel(3)(v1, v2) == PeriodicKernel(r = ones(3))(v1, v2)
    @test repr(k) == "Periodic Kernel, length(r) = $(length(r)))"
    test_ADs(r->PeriodicKernel(r =exp.(r)), log.(r), ADs = [:ForwardDiff, :ReverseDiff])
    @test_broken "Undefined adjoint for Sinus metric"
end
