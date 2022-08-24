# Tullio specialization for Euclidean and SqEuclidean metrics

function pairwise(::Euclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i, j] := sqrt <| (x.X[k, i] - y.X[k, j])^2
end

function ChainRulesCore.rrule(::typeof(pairwise), d::Euclidean, x::ColVecs, y::ColVecs)
    D = pairwise(d, x, y)
    function pairwise_pullback(Δ)
        @tullio ΔX[l, k] := Δ[k, i] * (x.X[l, k] - y.X[l, i]) / D[k, i]
        @tullio ΔY[l, i] := Δ[k, i] * (y.X[l, i] - x.X[l, k]) / D[k, i]
        return NoTangent(), NoTangent(), Tangent{ColVecs}(; X=ΔX), Tangent{ColVecs}(; X=ΔY)
    end
    return D, pairwise_pullback
end

function pairwise(::Euclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i, j] := sqrt <| (x.X[i, k] - y.X[j, k])^2
end

function ChainRulesCore.rrule(::typeof(pairwise), d::Euclidean, x::RowVecs, y::RowVecs)
    D = pairwise(d, x, y)
    function pairwise_pullback(Δ)
        @tullio ΔX[k, l] := Δ[k, i] * (x.X[k, l] - y.X[i, l]) / D[k, i]
        @tullio ΔY[i, l] := Δ[k, i] * (y.X[i, l] - x.X[k, l]) / D[k, i]
        return NoTangent(), NoTangent(), Tangent{RowVecs}(; X=ΔX), Tangent{RowVecs}(; X=ΔY)
    end
    return D, pairwise_pullback
end

function colwise(::Euclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i] := sqrt <| (x.X[k, i] - y.X[k, i])^2
end

function colwise(::Euclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i] := sqrt <| (x.X[i, k] - y.X[i, k])^2
end

function pairwise(::SqEuclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i, j] := (x.X[k, i] - y.X[k, j])^2
end

function pairwise(::SqEuclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i, j] := (x.X[i, k] - y.X[j, k])^2
end

function colwise(::SqEuclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i] := (x.X[k, i] - y.X[k, i])^2
end

function colwise(::SqEuclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i] := (x.X[i, k] - y.X[i, k])^2
end