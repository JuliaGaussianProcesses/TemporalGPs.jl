# All of this functionality is utilised only in the AD tests. Can be safely ignored if
# you're concerned with understanding how TemporalGPs works.

using ChainRulesCore: backing

# Functionality to test my testing functionality.
are_harmonised(a::Any, b::AbstractZero) = true
are_harmonised(a::AbstractZero, b::Any) = true
are_harmonised(a::AbstractZero, b::AbstractZero) = true

are_harmonised(a::Number, b::Number) = true

function are_harmonised(a::AbstractArray, b::AbstractArray)
    return all(ab -> are_harmonised(ab...), zip(a, b))
end

are_harmonised(a::Tuple, b::Tuple) = all(ab -> are_harmonised(ab...), zip(a, b))

function are_harmonised(a::Tangent{<:Any, <:Tuple}, b::Tangent{<:Any, <:Tuple})
    return all(ab -> are_harmonised(ab...), zip(a, b))
end

function are_harmonised(
    a::Tangent{<:Any, <:NamedTuple},
    b::Tangent{<:Any, <:NamedTuple},
)
    return all(
        name -> are_harmonised(getproperty(a, name), getproperty(b, name)),
        union(fieldnames(typeof(a)), fieldnames(typeof(b))),
    )
end

# Functionality to make it possible to compare different kinds of differentials. It's not
# entirely clear how much sense this makes mathematically, but it seems to work in a
# practical sense at the minute.
harmonise(a::Any, b::AbstractZero) = (a, b)
harmonise(a::AbstractZero, b::Any) = (a, b)
harmonise(a::AbstractZero, b::AbstractZero) = (a, b)

# Resolve ambiguity.
harmonise(a::AbstractZero, b::Tangent{<:Any, <:NamedTuple}) = (a, b)

harmonise(a::Number, b::Number) = (a, b)

function harmonise(a::Tuple, b::Tuple)
    vals = map(harmonise, a, b)
    return first.(vals), last.(vals)
end
function harmonise(a::AbstractArray, b::AbstractArray)
    vals = map(harmonise, a, b)
    return first.(vals), last.(vals)
end

function harmonise(a::Adjoint, b::Adjoint)
    vals = harmonise(a.parent, b.parent)
    return Tangent{Any}(parent=vals[1]), Tangent{Any}(parent=vals[2])
end

function harmonise(a::Tangent{<:Any, <:Tuple}, b::Tangent{<:Any, <:Tuple})
    vals = map(harmonise, backing(a), backing(b))
    return (Tangent{Any}(first.(vals)...), Tangent{Any}(last.(vals)...))
end

harmonise(a::Tangent{<:Any, <:Tuple}, b::Tuple) = harmonise(a, Tangent{Any}(b...))

harmonise(a::Tuple, b::Tangent{<:Any, <:Tuple}) = harmonise(Tangent{Any}(a...), b)

function harmonise(
    a::Tangent{<:Any, <:NamedTuple{names}},
    b::Tangent{<:Any, <:NamedTuple{names}},
) where {names}
    vals = map(harmonise, values(backing(a)), values(backing(b)))
    a_harmonised = Tangent{Any}(; NamedTuple{names}(first.(vals))...)
    b_harmonised = Tangent{Any}(; NamedTuple{names}(last.(vals))...)
    return (a_harmonised, b_harmonised)
end

function harmonise(a::Tangent{<:Any, <:NamedTuple}, b::Tangent{<:Any, <:NamedTuple})

    # Compute names missing / present in each data structure.
    a_names = propertynames(backing(a))
    b_names = propertynames(backing(b))
    mutual_names = intersect(a_names, b_names)
    all_names = (union(a_names, b_names)..., )
    a_missing_names = setdiff(all_names, a_names)
    b_missing_names = setdiff(all_names, b_names)

    # Construct `Tangent`s with the same names.
    a_vals = map(name -> name ∈ a_names ? getproperty(a, name) : ZeroTangent(), all_names)
    b_vals = map(name -> name ∈ b_names ? getproperty(b, name) : ZeroTangent(), all_names)
    a_unioned_names = Tangent{Any}(; NamedTuple{all_names}(a_vals)...)
    b_unioned_names = Tangent{Any}(; NamedTuple{all_names}(b_vals)...)

    # Harmonise those composites.
    return harmonise(a_unioned_names, b_unioned_names)
end

function harmonise(a::Tangent{<:Any, <:NamedTuple}, b)
    b_names = fieldnames(typeof(b))
    vals = map(name -> getfield(b, name), b_names)
    return harmonise(
        a, Tangent{Any}(; NamedTuple{b_names}(vals)...),
    )
end

harmonise(x::AbstractMatrix, y::NamedTuple{(:diag,)}) = (diag(x), y.diag)
function harmonise(x::AbstractVector, y::NamedTuple{(:value,:axes)})
    x = reduce(Zygote.accum, x)
    (x, y.value)
end


harmonise(a::Tangent{<:Any, <:NamedTuple}, b::AbstractZero) = (a, b)

harmonise(a, b::Tangent{<:Any, <:NamedTuple}) = reverse(harmonise(b, a))

# Special-cased handling for `Adjoint`s. Due to our usual AD setup, a differential for an
# Adjoint can be represented either by a matrix or a `Tangent`. Both ought to `to_vec` to
# the same thing though, so this should be fine for now, if a little unsatisfactory.
function harmonise(a::Adjoint, b::Tangent{<:Adjoint, <:NamedTuple})
    return Tangent{Any}(parent=parent(a)), b
end
