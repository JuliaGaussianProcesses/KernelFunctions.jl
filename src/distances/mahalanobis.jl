Distances.pairwise(d::SqMahalanobis, x::ColVecs, y::ColVecs) = pairwise(d, x.X, y.X; dims=2)
Distances.pairwise(d::SqMahalanobis, x::RowVecs, y::RowVecs) = pairwise(d, x.X, y.X; dims=1)
