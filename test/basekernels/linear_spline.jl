@testset "linear_spline" begin
    test_interface(StableRNG(123456), LinearSplineKernel(1.1), Vector{Float64})
end
