function kernel_sum end
const ⊕ = kernel_sum

for (M, op, T) in (
    (:Base, :+, :KernelSum),
    (:Base, :*, :KernelProduct),
    (:TensorCore, :tensor, :KernelTensorProduct),
    (:KernelFunctions, :⊕, :KernelTensorSum),
)
    @eval begin
        $M.$op(k1::Kernel, k2::Kernel) = $T(k1, k2)

        $M.$op(k1::$T, k2::$T) = $T(k1.kernels..., k2.kernels...)
        function $M.$op(
            k1::$T{<:AbstractVector{<:Kernel}}, k2::$T{<:AbstractVector{<:Kernel}}
        )
            return $T(vcat(k1.kernels, k2.kernels))
        end

        $M.$op(k::Kernel, ks::$T) = $T(k, ks.kernels...)
        $M.$op(k::Kernel, ks::$T{<:AbstractVector{<:Kernel}}) = $T(vcat(k, ks.kernels))

        $M.$op(ks::$T, k::Kernel) = $T(ks.kernels..., k)
        $M.$op(ks::$T{<:AbstractVector{<:Kernel}}, k::Kernel) = $T(vcat(ks.kernels, k))

        # Fix method ambiguity issues
        function $M.$op(ks1::$T, ks2::$T{<:AbstractVector{<:Kernel}})
            return $T(vcat(collect(ks1.kernels), ks2.kernels))
        end
        function $M.$op(ks1::$T{<:AbstractVector{<:Kernel}}, ks2::$T)
            return $T(vcat(ks1.kernels, collect(ks2.kernels)))
        end
    end
end
