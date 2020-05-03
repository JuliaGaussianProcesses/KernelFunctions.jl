"""
    StretchTransform()


"""
struct StretchTransform{M} <: Transform where M<:AbstractMatrix{<:Real}
    w::M
    function StretchTransform(w)
        @assert all(w .>= 0) "Invalid Weights"
        new{typeof(w)}(w)
    end
end

Base.size(kernel::StretchTransform) = size(kernel.w)

(t::StretchTransform)(x) = t.w' * x

Base.show(io::IO, t::StretchTransform) = print(io, "Stretch Transform (D = ", size(t.w, 1),
                                               ", K = ", size(t.w, 2), ")")

