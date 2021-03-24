ZygoteRules.@adjoint function Base.map(t::Transform, X::ColVecs)
    return ZygoteRules.pullback(_map, t, X)
end

ZygoteRules.@adjoint function Base.map(t::Transform, X::RowVecs)
    return ZygoteRules.pullback(_map, t, X)
end
