#ZygoteRules.@adjoint function Base.map(t::Transform, X::ColVecs)
#    return ZygoteRules.pullback(_map, t, X)
#end
#
#ZygoteRules.@adjoint function Base.map(t::Transform, X::RowVecs)
#    return ZygoteRules.pullback(_map, t, X)
#end
#
#function ZygoteRules._pullback(
#    cx::AContext, ::typeof(literal_getproperty), x::ColVecs, ::Val{f}
#) where {f}
#    return ZygoteRules._pullback(cx, literal_getfield, x, Val{f}())
#end
