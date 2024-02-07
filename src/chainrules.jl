## Forward Rules

# Note that this is type piracy as the derivative should be NaN for x == y.
function ChainRulesCore.frule(
    (_, Δx, Δy)::Tuple{<:Any,<:Any,<:Any},
    d::Distances.Euclidean,
    x::AbstractVector,
    y::AbstractVector,
)
    Δ = x - y
    D = sqrt(sum(abs2, Δ))
    if !iszero(D)
        Δ ./= D
    end
    return D, dot(Δ, Δx) - dot(Δ, Δy)
end

## Reverse Rules Delta

function ChainRulesCore.rrule(dist::Delta, x::AbstractVector, y::AbstractVector)
    d = dist(x, y)
    function evaluate_pullback(::Any)
        return NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return d, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Delta, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent()
    end
    return P, pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::Delta, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(::Any)
        return NoTangent(), NoTangent(), ZeroTangent(), ZeroTangent()
    end
    return C, colwise_pullback
end

## Reverse Rules DotProduct

function ChainRulesCore.rrule(dist::DotProduct, x::AbstractVector, y::AbstractVector)
    d = dist(x, y)
    function evaluate_pullback(Δ::Any)
        return NoTangent(), Δ .* y, Δ .* x
    end
    return d, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise),
    d::DotProduct,
    X::AbstractMatrix,
    Y::AbstractMatrix;
    dims=2,
)
    P = Distances.pairwise(d, X, Y; dims=dims)
    function pairwise_pullback_cols(Δ::Any)
        if dims == 1
            return NoTangent(), NoTangent(), Δ * Y, Δ' * X
        else
            return NoTangent(), NoTangent(), Y * Δ', X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::DotProduct, X::AbstractMatrix; dims=2
)
    P = Distances.pairwise(d, X; dims=dims)
    function pairwise_pullback_cols(Δ::Any)
        if dims == 1
            return NoTangent(), NoTangent(), 2 * Δ * X
        else
            return NoTangent(), NoTangent(), 2 * X * Δ
        end
    end
    return P, pairwise_pullback_cols
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::DotProduct, X::AbstractMatrix, Y::AbstractMatrix
)
    C = Distances.colwise(d, X, Y)
    function colwise_pullback(Δ::Any)
        return NoTangent(), NoTangent(), Δ' .* Y, Δ' .* X
    end
    return C, colwise_pullback
end

## Reverse Rules Sinus

function ChainRulesCore.rrule(s::Sinus, x::AbstractVector, y::AbstractVector)
    d = x - y
    abs2_sind_r = (sinpi.(d) ./ s.r) .^ 2
    val = sum(abs2_sind_r)
    gradx = π .* cospi.(2 .* d) ./ s.r .^ 2
    function evaluate_pullback(Δ::Any)
        r̄ = -2Δ .* abs2_sind_r ./ s.r
        s̄ = ChainRulesCore.Tangent{typeof(s)}(; r=r̄)
        return s̄, Δ * gradx, -Δ * gradx
    end
    return val, evaluate_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Sinus, x::AbstractMatrix; dims=2
)
    project_x = ProjectTo(x)
    function pairwise_pullback(z̄)
        Δ = unthunk(z̄)
        n = size(x, dims)
        x̄ = collect(zero(x))
        r̄ = zero(d.r)
        if dims == 1
            for j in 1:n, i in 1:n
                xi = view(x, i, :)
                xj = view(x, j, :)
                ds = π .* Δ[i, j] .* cospi.(2 .* (xi .- xj)) ./ d.r .^ 2
                r̄ .-= 2 .* Δ[i, j] .* sinpi.(xi .- xj) .^ 2 ./ d.r .^ 3
                x̄[i, :] += ds
                x̄[j, :] -= ds
            end
        elseif dims == 2
            for j in 1:n, i in 1:n
                xi = view(x, :, i)
                xj = view(x, :, j)
                ds = twoπ .* Δ[i, j] .* sinpi.(xi .- xj) .* cospi.(xi .- xj) ./ d.r .^ 2
                r̄ .-= 2 .* Δ[i, j] .* sinpi.(xi .- xj) .^ 2 ./ d.r .^ 3
                x̄[:, i] .+= ds
                x̄[:, j] .-= ds
            end
        end
        d̄ = ChainRulesCore.Tangent{typeof(d)}(; r=r̄)
        return NoTangent(), d̄, @thunk(project_x(x̄))
    end
    return Distances.pairwise(d, x; dims), pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.pairwise), d::Sinus, x::AbstractMatrix, y::AbstractMatrix; dims=2
)
    project_x = ProjectTo(x)
    project_y = ProjectTo(y)
    function pairwise_pullback(z̄)
        Δ = unthunk(z̄)
        n = size(x, dims)
        m = size(y, dims)
        x̄ = collect(zero(x))
        ȳ = collect(zero(y))
        r̄ = zero(d.r)
        if dims == 1
            for j in 1:m, i in 1:n
                xi = view(x, i, :)
                yj = view(y, j, :)
                ds = π .* Δ[i, j] .* cospi.(2 .* (xi .- yj)) ./ d.r .^ 2
                r̄ .-= 2 .* Δ[i, j] .* sinpi.(xi .- yj) .^ 2 ./ d.r .^ 3
                x̄[i, :] .+= ds
                ȳ[j, :] .-= ds
            end
        elseif dims == 2
            for j in 1:m, i in 1:n
                xi = view(x, :, i)
                yj = view(y, :, j)
                ds = π .* Δ[i, j] .* cospi.(2 .* (xi .- yj)) ./ d.r .^ 2
                r̄ .-= 2 .* Δ[i, j] .* sinpi.(xi .- yj) .^ 2 ./ d.r .^ 3
                x̄[:, i] .+= ds
                ȳ[:, j] .-= ds
            end
        end
        d̄ = ChainRulesCore.Tangent{typeof(d)}(; r=r̄)
        return NoTangent(), d̄, @thunk(project_x(x̄)), @thunk(project_y(ȳ))
    end
    return Distances.pairwise(d, x, y; dims), pairwise_pullback
end

function ChainRulesCore.rrule(
    ::typeof(Distances.colwise), d::Sinus, x::AbstractMatrix, y::AbstractMatrix
)
    project_x = ProjectTo(x)
    project_y = ProjectTo(y)
    function colwise_pullback(z̄)
        Δ = unthunk(z̄)
        n = size(x, 2)
        x̄ = collect(zero(x))
        ȳ = collect(zero(y))
        r̄ = zero(d.r)
        for i in 1:n
            xi = view(x, :, i)
            yi = view(y, :, i)
            ds = π .* Δ[i] .* cospi.(2 .* (xi .- yi)) ./ d.r .^ 2
            r̄ .-= 2 .* Δ[i] .* sinpi.(xi .- yi) .^ 2 ./ d.r .^ 3
            x̄[:, i] .+= ds
            ȳ[:, i] .-= ds
        end
        d̄ = ChainRulesCore.Tangent{typeof(d)}(; r=r̄)
        return NoTangent(), d̄, @thunk(project_x(x̄)), @thunk(project_y(ȳ))
    end
    return Distances.colwise(d, x, y), colwise_pullback
end

## Reverse Rules for matrix wrappers

function ChainRulesCore.rrule(::Type{<:ColVecs}, X::AbstractMatrix)
    ColVecs_pullback(Δ::Tangent) = (NoTangent(), Δ.X)
    function ColVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return error(
            "Pullback on AbstractVector{<:AbstractVector}.\n" *
            "This might happen if you try to use gradients on the generic `kernelmatrix` or `kernelmatrix_diag`,\n" *
            "or because some external computation has acted on `ColVecs` to produce a vector of vectors." *
            "In the former case, to solve this issue overload `kernelmatrix(_diag)` for your kernel for `ColVecs`." *
            "In the latter case, one needs to track down the `rrule` whose pullback returns a `Vector{Vector{T}}`," *
            " rather than a `Tangent`, as the cotangent / gradient for `ColVecs` input, and circumvent it."
        )
    end
    return ColVecs(X), ColVecs_pullback
end

function ChainRulesCore.rrule(::Type{<:RowVecs}, X::AbstractMatrix)
    RowVecs_pullback(Δ::Tangent) = (NoTangent(), Δ.X)
    function RowVecs_pullback(::AbstractVector{<:AbstractVector{<:Real}})
        return error(
            "Pullback on AbstractVector{<:AbstractVector}.\n" *
            "This might happen if you try to use gradients on the generic `kernelmatrix` or `kernelmatrix_diag`,\n" *
            "or because some external computation has acted on `RowVecs` to produce a vector of vectors." *
            "If it is the former, to solve this issue overload `kernelmatrix(_diag)` for your kernel for `RowVecs`",
        )
    end
    return RowVecs(X), RowVecs_pullback
end
