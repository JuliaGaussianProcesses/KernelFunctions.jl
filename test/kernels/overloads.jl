@testset "overloads" begin
    rng = MersenneTwister(123456)

    k1 = LinearKernel()
    k2 = SqExponentialKernel()
    k3 = RationalQuadraticKernel()

    for (op, T) in ((+, KernelSum), (*, KernelProduct), (⊗, KernelTensorProduct))
        if T === KernelTensorProduct
            v2_1 = rand(rng, 2)
            v2_2 = rand(rng, 2)
            v3_1 = rand(rng, 3)
            v3_2 = rand(rng, 3)
            v4_1 = rand(rng, 4)
            v4_2 = rand(rng, 4)
        else
            v2_1 = v3_1 = v4_1 = rand(rng, 3)
            v2_2 = v3_2 = v4_2 = rand(rng, 3)
        end
        k = T(k1, k2)

        @test op(k1, k2)(v2_1, v2_2) == k(v2_1, v2_2)
        @test op(k, k3)(v3_1, v3_2) == T((k1, k2, k3))(v3_1, v3_2)
        @test op(k3, k)(v3_1, v3_2) == T((k3, k1, k2))(v3_1, v3_2)
        @test op(k, k)(v4_1, v4_2) == T((k1, k2, k1, k2))(v4_1, v4_2)
        @test op(k1, k2) == T([k1, k2]) == T((k1, k2))

        @test op(T([k1, k2]), T([k2, k1])).kernels == [k1, k2, k2, k1]
        @test op(T([k1, k2]), k3).kernels == [k1, k2, k3]
        @test op(k3, T([k1, k2])).kernels == [k3, k1, k2]

        @test op(T((k1, k2)), T((k2, k1))).kernels == (k1, k2, k2, k1)
        @test op(T((k1, k2)), k3).kernels == (k1, k2, k3)
        @test op(k3, T((k1, k2))).kernels == (k3, k1, k2)
    end
end
