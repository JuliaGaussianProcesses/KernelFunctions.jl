@testset "linear_spline" begin
    test_interface(StableRNG(123456), LinearSplineKernel(1.1), Vector{Float64})
    test_interface_ad_perf(LinearSplineKernel, 1.0, StableRNG(123456), [Vector{Float64}])
end
