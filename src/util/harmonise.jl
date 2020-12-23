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

function are_harmonised(a::Composite{<:Any, <:Tuple}, b::Composite{<:Any, <:Tuple})
    return all(ab -> are_harmonised(ab...), zip(a, b))
end

function are_harmonised(
    a::Composite{<:Any, <:NamedTuple},
    b::Composite{<:Any, <:NamedTuple},
)
    return all(
        name -> are_harmonised(getproperty(a, name), getproperty(b, name)),
        union(propertynames(a), propertynames(b)),
    )
end

# Functionality to make it possible to compare different kinds of differentials. It's not
# entirely clear how much sense this makes mathematically, but it seems to work in a
# practical sense at the minute.
harmonise(a::Any, b::AbstractZero) = (a, b)
harmonise(a::AbstractZero, b::Any) = (a, b)
harmonise(a::AbstractZero, b::AbstractZero) = (a, b)

# Resolve ambiguity.
harmonise(a::AbstractZero, b::Composite{<:Any, <:NamedTuple}) = (a, b)

harmonise(a::Number, b::Number) = (a, b)

function harmonise(a::Tuple, b::Tuple)
    vals = map(harmonise, a, b)
    return first.(vals), last.(vals)
end
function harmonise(a::AbstractArray, b::AbstractArray)
    vals = map(harmonise, a, b)
    return first.(vals), last.(vals)
end

function harmonise(a::Composite{<:Any, <:Tuple}, b::Composite{<:Any, <:Tuple})
    vals = map(harmonise, backing(a), backing(b))
    return (Composite{Any}(first.(vals)...), Composite{Any}(last.(vals)...))
end

harmonise(a::Composite{<:Any, <:Tuple}, b::Tuple) = harmonise(a, Composite{Any}(b...))

harmonise(a::Tuple, b::Composite{<:Any, <:Tuple}) = harmonise(Composite{Any}(a...), b)

function harmonise(
    a::Composite{<:Any, <:NamedTuple{names}},
    b::Composite{<:Any, <:NamedTuple{names}},
) where {names}
    vals = map(harmonise, values(backing(a)), values(backing(b)))
    a_harmonised = Composite{Any}(; NamedTuple{names}(first.(vals))...)
    b_harmonised = Composite{Any}(; NamedTuple{names}(last.(vals))...)
    return (a_harmonised, b_harmonised)
end

function harmonise(a::Composite{<:Any, <:NamedTuple}, b::Composite{<:Any, <:NamedTuple})

    # Compute names missing / present in each data structure.
    a_names = propertynames(backing(a))
    b_names = propertynames(backing(b))
    mutual_names = intersect(a_names, b_names)
    all_names = (union(a_names, b_names)..., )
    a_missing_names = setdiff(all_names, a_names)
    b_missing_names = setdiff(all_names, b_names)

    # Construct `Composite`s with the same names.
    a_vals = map(name -> name ∈ a_names ? getproperty(a, name) : Zero(), all_names)
    b_vals = map(name -> name ∈ b_names ? getproperty(b, name) : Zero(), all_names)
    a_unioned_names = Composite{Any}(; NamedTuple{all_names}(a_vals)...)
    b_unioned_names = Composite{Any}(; NamedTuple{all_names}(b_vals)...)

    # Harmonise those composites.
    return harmonise(a_unioned_names, b_unioned_names)
end

function harmonise(a::Composite{<:Any, <:NamedTuple}, b)
    b_names = propertynames(b)
    vals = map(name -> getproperty(b, name), b_names)
    return harmonise(
        a, Composite{Any}(; NamedTuple{b_names}(vals)...),
    )
end

harmonise(a, b::Composite{<:Any, <:NamedTuple}) = reverse(harmonise(b, a))
