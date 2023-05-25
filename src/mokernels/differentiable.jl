
struct Partial{Order}
	indices::CartesianIndex{Order}
end

function Partial(indices::Integer...)
	return Partial{length(indices)}(CartesianIndex(indices))
end

compact_string_representation(::Partial{0}) = print(io, "id")
function compact_string_representation(p::Partial)
	tuple = Tuple(p.indices)
	lower_numbers = @. tuple |> digits |> reverse |> n-> '₀' + n
	return join(["∂$(join(x))" for x in lower_numbers])
end
function Base.show(io::IO, p::Partial)
	if get(io, :compact, false)
		print(io, "Partial($(Tuple(p.indices)))")	
	else
		print(io, compact_string_representation(p)) 
	end
end

function Base.show(io::IO, ::MIME"text/html", p::Partial)
	tuple = Tuple(p.indices)
	if get(io, :compact, false)
		print(io, join(map(n->"∂<sub>$(n)</sub>", tuple),""))
	else
		print(io, compact_string_representation(p))
	end
end