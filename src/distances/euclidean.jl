# Tullio specialization for Euclidean and SqEuclidean metrics

function pairwise(::Euclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i, j] := sqrt <| x.X[k, i] ^ 2 - 2 * x.X[k, i] * y.X[k, j] + y.X[k, j] ^ 2
end

function pairwise(::Euclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i, j] := sqrt <| x.X[i, k] ^ 2 - 2 * x.X[i, k] * y.X[j, k] + y.X[j, k] ^ 2
end

function pairwise(::SqEuclidean, x::ColVecs, y::ColVecs)
    return @tullio out[i, j] := x.X[k, i] ^ 2 - 2 * x.X[k, i] * y.X[k, j] + y.X[k, j] ^ 2
end

function pairwise(::SqEuclidean, x::RowVecs, y::RowVecs)
    return @tullio out[i, j] := x.X[i, k] ^ 2 - 2 * x.X[i, k] * y.X[j, k] + y.X[j, k] ^ 2
end