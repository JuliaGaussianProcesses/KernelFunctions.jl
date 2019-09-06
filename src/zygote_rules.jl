using ZygoteRules

@adjoint function colwise(s::Euclidean, x::AbstractMatrix, y::AbstractMatrix)
    d = colwise(s, x, y)
    return d, function (Δ::AbstractVector)
        x̄ = (Δ ./ d)' .* (x .- y)
        return nothing, x̄, -x̄
    end
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix, Y::AbstractMatrix; dims=2)
    @assert dims == 2
    D, back = Zygote.forward((X, Y)->pairwise(SqEuclidean(), X, Y; dims=2), X, Y)
    D .= sqrt.(D)
    return D, Δ -> (nothing, back(Δ ./ (2 .* D))...)
end

@adjoint function pairwise(::Euclidean, X::AbstractMatrix; dims=2)
    @assert dims == 2
    D, back = Zygote.forward(X->pairwise(SqEuclidean(), X; dims=2), X)
    D .= sqrt.(D)
    return D, function(Δ)
        Δ = Δ ./ (2 .* D)
        Δ[diagind(Δ)] .= 0
        return (nothing, first(back(Δ)))
    end
end
