function FiniteDifferences.to_vec(x::Tuple{T, Int}) where {T}
    function MOinput_from_vec(x_vec)
        return first(x_vec)
    end
    return [x], MOinput_from_vec
end

FiniteDifferences.to_vec(x::Vector{Tuple{T, Int}}) where {T} = (x, identity)

function FiniteDifferences._j′vp(fdm, f, ȳ::Vector{<:Real}, x::Vector{Tuple{T, Int}}) where {T}
    isempty(x) && return eltype(ȳ)[] # if x is empty, then so is the jacobian and x̄
    return transpose(first(jacobian(fdm, f, x))) * ȳ
end

function FiniteDifferences.jacobian(fdm, f, x::Vector{Tuple{T, Int}}; len=nothing) where {T}
    len !== nothing && Base.depwarn(
        "`len` keyword argument to `jacobian` is no longer required " *
        "and will not be permitted in the future.",
         :jacobian
    )
    ẏs = map(eachindex(x)) do n
        return fdm(zero(eltype(x).types[1])) do ε
            xn = x[n]
            x[n] = (xn[1] + ε, xn[2])
            ret = copy(first(to_vec(f(x))))  # copy required incase `f(x)` returns something that aliases `x`
            x[n] = xn  # Can't do `x[n] -= ϵ` as floating-point math is not associative
            return ret
        end
    end
    return (hcat(ẏs...), )
end
