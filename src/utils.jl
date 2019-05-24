## Macro for checking
macro check_args(K, param, cond, desc=string(cond))
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(K)), ": ", $(string(param)), " = ", $(esc(param)), " does not ",
                "satisfy the constraint ", $(string(desc)), ".")))
        end
    end
end

function promote_float(Tₖ::DataType...)
    if length(Tₖ) == 0
        return Float64
    end
    T = promote_type(Tₖ...)
    return T <: Real ? T : Float64
end
